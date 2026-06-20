"""
DRL environment design (Section 9 of implementation_v2).

State space (compact per-path, per-core features - not raw occupancy matrices):
  for each candidate route and each core: the start index and size of the first
  available slot block, the total available slots, the mean block size, and the
  DC slots required on that route. Plus a request descriptor: source/destination
  (one-hot) and the three channel requirements.

Action space (a single flattened discrete action, factored as):
  * route selection      - which of the K candidate routes,
  * quantum-core selection - which core in C_Q,
  * spectrum fit         - first / best / random fit for QC, CC and DC.

Action masking: infeasible (route, core) options - those that cannot host the
quantum channel under the XT-avoided mask and the spectrum constraints - are
zeroed before the policy output, so every executed allocation is feasible.

Reward: +1 for a fully served triplet (QC, CC and DC), -1 for a blocked request.
"""

import heapq
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

from allocator import allocate, qc_route_core_feasible
from config import Config
from modulation import ModulationSelector
from spectrum import Allocation, SpectrumState
from topology import Path, Topology
from traffic import Request, TrafficGenerator

FEATURES_PER_PATH_CORE = 5      # start, size, total_avail, mean_block, dc_slots


class QKDSSEONEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.topo = Topology(cfg.net.topology, cfg.net.num_cores,
                             cfg.xt.adjacency_layout)
        self.spectrum = SpectrumState(self.topo, cfg.net.slots_per_core,
                                      cfg.alloc.guard_band, cfg.seed)
        self.modsel = ModulationSelector(cfg.mod)
        self.traffic = TrafficGenerator(
            self.topo.num_nodes, cfg.req.capacity_classes,
            cfg.req.arrival_rate, cfg.req.mean_holding_time, cfg.seed)

        self.K = cfg.alloc.k_paths
        self.n_cores = cfg.net.num_cores
        self.n_qcores = len(cfg.net.quantum_cores)
        self.n_fits = len(cfg.drl.fit_strategies)

        # Flattened action factors: (route, q_core, fit_qc, fit_cc, fit_dc).
        self._action_factors = (self.K, self.n_qcores,
                                self.n_fits, self.n_fits, self.n_fits)
        self.action_dim = int(np.prod(self._action_factors))
        self.obs_dim = (self.K * self.n_cores * FEATURES_PER_PATH_CORE
                        + 2 * self.topo.num_nodes + 3)

        # Active lightpaths, as a heap keyed by departure time.
        self._active: List[Tuple[float, int, List[Allocation]]] = []
        self._counter = itertools.count()  # tie-breaker for the heap
        self.current_request: Optional[Request] = None
        self._candidate_routes: List[Path] = []
        self._served = 0
        self._blocked = 0
        self._step_in_episode = 0

    # ---- action (de)coding ---------------------------------------------------
    def decode_action(self, flat: int) -> Tuple[int, int, str, str, str]:
        route, q_idx, f_qc, f_cc, f_dc = np.unravel_index(
            flat, self._action_factors)
        fits = self.cfg.drl.fit_strategies
        q_core = self.cfg.net.quantum_cores[int(q_idx)]
        return (int(route), q_core, fits[int(f_qc)], fits[int(f_cc)], fits[int(f_dc)])

    def action_mask(self) -> np.ndarray:
        """Boolean mask over the flattened action space. A (route, q_core)
        pair is feasible iff a quantum channel could be placed there."""
        mask = np.zeros(self.action_dim, dtype=bool)
        feasible_rc = np.zeros((self.K, self.n_qcores), dtype=bool)
        for r in range(self.K):
            if r >= len(self._candidate_routes):
                continue
            route = self._candidate_routes[r]
            for qi, q_core in enumerate(self.cfg.net.quantum_cores):
                feasible_rc[r, qi] = qc_route_core_feasible(
                    self.spectrum, self.topo, route, q_core, self.cfg)
        # Expand to all fit combinations.
        for r in range(self.K):
            for qi in range(self.n_qcores):
                if not feasible_rc[r, qi]:
                    continue
                for f1 in range(self.n_fits):
                    for f2 in range(self.n_fits):
                        for f3 in range(self.n_fits):
                            flat = np.ravel_multi_index(
                                (r, qi, f1, f2, f3), self._action_factors)
                            mask[flat] = True
        return mask

    # ---- observation ---------------------------------------------------------
    def _path_core_features(self, route: Optional[Path], core: int,
                            dc_slots: int) -> List[float]:
        S = self.cfg.net.slots_per_core
        if route is None:
            return [-1.0] * FEATURES_PER_PATH_CORE
        avail = self.spectrum.availability_mask(route, core)
        runs = SpectrumState._run_lengths(avail)
        # first available block
        first_start, first_size = -1, 0
        idx = np.flatnonzero(avail)
        if idx.size > 0:
            first_start = int(idx[0])
            first_size = int(runs[first_start])
        total_avail = int(avail.sum())
        # mean block size over the free blocks
        block_sizes = [runs[i] for i in range(S)
                       if avail[i] and (i == 0 or not avail[i - 1])]
        mean_block = float(np.mean(block_sizes)) if block_sizes else 0.0
        return [first_start / S, first_size / S, total_avail / S,
                mean_block / S, dc_slots / S]

    def _observation(self) -> np.ndarray:
        S = self.cfg.net.slots_per_core
        feats: List[float] = []
        for r in range(self.K):
            route = self._candidate_routes[r] if r < len(self._candidate_routes) else None
            if route is not None:
                sel = self.modsel.select(self.topo.path_length(route),
                                         self.current_request.demand)
                dc_slots = sel[2] if sel else 0
            else:
                dc_slots = 0
            for c in range(self.n_cores):
                feats.extend(self._path_core_features(route, c, dc_slots))

        # request descriptor: src/dst one-hot + three channel requirements
        src_oh = np.zeros(self.topo.num_nodes)
        dst_oh = np.zeros(self.topo.num_nodes)
        src_oh[self.current_request.src] = 1.0
        dst_oh[self.current_request.dst] = 1.0
        # DC requirement taken on the shortest candidate route.
        if self._candidate_routes:
            sel = self.modsel.select(
                self.topo.path_length(self._candidate_routes[0]),
                self.current_request.demand)
            dc_req = (sel[2] / S) if sel else 0.0
        else:
            dc_req = 0.0
        descriptor = [1.0 / S, 1.0 / S, dc_req]      # QC, CC, DC requirements
        return np.concatenate([np.asarray(feats, dtype=np.float32),
                               src_oh, dst_oh,
                               np.asarray(descriptor, dtype=np.float32)]).astype(np.float32)

    # ---- simulation bookkeeping ---------------------------------------------
    def _release_expired(self, now: float):
        while self._active and self._active[0][0] <= now:
            _, _, allocs = heapq.heappop(self._active)
            self.spectrum.release_all(allocs)

    def _fetch_next_request(self):
        self.current_request = self.traffic.next_request()
        self._release_expired(self.current_request.arrival_time)
        self._candidate_routes = list(self.topo.k_shortest_paths(
            self.current_request.src, self.current_request.dst, self.K))

    # ---- gym-style API -------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.cfg.seed = seed
        self.spectrum.reset(seed)
        self.traffic.reset(seed)
        self._active = []
        self._served = 0
        self._blocked = 0
        self._step_in_episode = 0
        self._fetch_next_request()
        return self._observation()

    def step(self, flat_action: int):
        route_idx, q_core, f_qc, f_cc, f_dc = self.decode_action(flat_action)
        result = allocate(self.spectrum, self.topo, self.modsel, self.cfg,
                          self.current_request, self._candidate_routes,
                          route_idx, q_core, f_qc, f_cc, f_dc)

        if result.success:
            reward = 1.0
            self._served += 1
            heapq.heappush(self._active,
                           (self.current_request.departure_time,
                            next(self._counter), result.allocations))
        else:
            reward = -1.0
            self._blocked += 1

        self._step_in_episode += 1
        done = self._step_in_episode >= self.cfg.drl.episode_length

        info = {
            "success": result.success,
            "blocked_reason": result.blocked_reason,
            "utilization": self.spectrum.utilization(),
            "fragmentation": self.spectrum.external_fragmentation(),
        }

        self._fetch_next_request()        # advance to the next arrival
        return self._observation(), reward, done, info

    # ---- metrics (Section 11) -----------------------------------------------
    @property
    def blocking_probability(self) -> float:
        total = self._served + self._blocked
        return self._blocked / total if total else 0.0

    @property
    def triplet_success_rate(self) -> float:
        total = self._served + self._blocked
        return self._served / total if total else 0.0
