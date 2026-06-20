"""
Spectrum resource state and allocation primitives
(Sections 5 & 6 of implementation_v2).

Constraints enforced here:
  * Spectrum continuity  - the same slot indices on every link of the route.
  * Spectrum contiguity  - allocated slots are adjacent.
  * Core continuity      - the same core along the whole route.
  * Crosstalk avoidance  - the XT-avoided binary mask: a slot on core c (link e)
                           may be used only if that slot is free on every core
                           adjacent to c on link e.
  * Guard band           - 1 free slot between any two adjacent allocations on
                           the same core.

Crosstalk is a binary feasibility mask, not a continuous quantity (Section 6):
adjacent-core occupancy simply makes a slot infeasible.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from topology import Edge, Path, Topology


@dataclass
class Allocation:
    """A committed slice of spectrum for one channel."""
    channel: str                # "QC", "CC" or "DC"
    path: Path
    core: int
    start: int
    num_slots: int


class SpectrumState:
    def __init__(self, topology: Topology, slots_per_core: int,
                 guard_band: int = 1, seed: int = 0):
        self.topo = topology
        self.S = slots_per_core
        self.guard = guard_band
        self.rng = np.random.default_rng(seed)
        # occupancy[edge, core, slot] = True if used.
        self.occupancy = np.zeros(
            (topology.num_edges, topology.num_cores, self.S), dtype=bool)

    # ---- helpers -------------------------------------------------------------
    def _edge_ids(self, path: Path) -> List[int]:
        return [self.topo.edge_index[e] for e in path]

    def _core_free_on_path(self, edge_ids, core) -> np.ndarray:
        """Slots free on `core` across every link of the path (continuity)."""
        return ~np.any(self.occupancy[edge_ids, core, :], axis=0)

    def _xt_free_on_path(self, edge_ids, core) -> np.ndarray:
        """Slots free on every adjacent core across every link (XT-avoided)."""
        adj = self.topo.adjacent_cores(core)
        if not adj:
            return np.ones(self.S, dtype=bool)
        # free on all adjacent cores and all path links
        return ~np.any(self.occupancy[np.ix_(edge_ids, adj)], axis=(0, 1))

    @staticmethod
    def _run_lengths(mask: np.ndarray) -> np.ndarray:
        """For each index, the length of the maximal True-run it belongs to."""
        out = np.zeros(mask.shape[0], dtype=int)
        i = 0
        n = mask.shape[0]
        while i < n:
            if mask[i]:
                j = i
                while j < n and mask[j]:
                    j += 1
                out[i:j] = j - i
                i = j
            else:
                i += 1
        return out

    # ---- feasibility / search -----------------------------------------------
    def candidate_starts(self, path: Path, core: int, n_slots: int) -> List[int]:
        """All start indices where `n_slots` can be placed on `core` along
        `path`, honouring continuity, contiguity, the guard band and the
        XT-avoided mask."""
        if n_slots <= 0 or n_slots > self.S:
            return []
        edge_ids = self._edge_ids(path)
        core_free = self._core_free_on_path(edge_ids, core)
        avail = core_free & self._xt_free_on_path(edge_ids, core)

        starts: List[int] = []
        # cumulative count of available slots for an O(1) window-sum check.
        csum = np.concatenate([[0], np.cumsum(avail)])
        for s in range(0, self.S - n_slots + 1):
            if csum[s + n_slots] - csum[s] != n_slots:
                continue                                   # block not fully free
            if s > 0 and not core_free[s - 1]:
                continue                                   # left guard (own core)
            if s + n_slots < self.S and not core_free[s + n_slots]:
                continue                                   # right guard (own core)
            starts.append(s)
        return starts

    def select_start(self, path: Path, core: int, n_slots: int,
                     fit: str) -> Optional[int]:
        """Choose a start slot using a first-fit, best-fit or random-fit policy.

        The DRL agent selects the fit strategy per channel (Section 9)."""
        starts = self.candidate_starts(path, core, n_slots)
        if not starts:
            return None
        if fit == "first":
            return starts[0]
        if fit == "random":
            return int(self.rng.choice(starts))
        if fit == "best":
            # best-fit: the start whose enclosing free run is smallest.
            edge_ids = self._edge_ids(path)
            avail = (self._core_free_on_path(edge_ids, core)
                     & self._xt_free_on_path(edge_ids, core))
            runs = self._run_lengths(avail)
            return min(starts, key=lambda s: (runs[s], s))
        raise ValueError(f"unknown fit strategy '{fit}'")

    def can_place(self, path: Path, core: int, n_slots: int) -> bool:
        return len(self.candidate_starts(path, core, n_slots)) > 0

    def availability_mask(self, path: Path, core: int) -> np.ndarray:
        """Combined own-core + XT-avoided free mask along a path. This is the
        set of slots usable by a new connection on (path, core)."""
        edge_ids = self._edge_ids(path)
        return self._core_free_on_path(edge_ids, core) & self._xt_free_on_path(edge_ids, core)

    # ---- commit / release ----------------------------------------------------
    def commit(self, path: Path, core: int, start: int, n_slots: int,
               channel: str) -> Allocation:
        edge_ids = self._edge_ids(path)
        self.occupancy[edge_ids, core, start:start + n_slots] = True
        return Allocation(channel, list(path), core, start, n_slots)

    def release(self, alloc: Allocation):
        edge_ids = self._edge_ids(alloc.path)
        self.occupancy[edge_ids, alloc.core,
                       alloc.start:alloc.start + alloc.num_slots] = False

    def release_all(self, allocs: List[Allocation]):
        for a in allocs:
            self.release(a)

    # ---- metrics (Section 8 & 11) -------------------------------------------
    def utilization(self) -> float:
        return float(self.occupancy.mean())

    def external_fragmentation(self) -> float:
        """Mean external fragmentation ratio over all (link, core) fibers:
        1 - (largest free block / total free slots)."""
        ratios = []
        for e in range(self.topo.num_edges):
            for c in range(self.topo.num_cores):
                free = ~self.occupancy[e, c, :]
                total_free = int(free.sum())
                if total_free == 0:
                    continue
                largest = int(self._run_lengths(free).max(initial=0))
                ratios.append(1.0 - largest / total_free)
        return float(np.mean(ratios)) if ratios else 0.0

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.occupancy.fill(False)
