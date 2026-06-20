"""
Per-request allocation procedure (Section 7 of implementation_v2).

Allocate the three channels in order; if any one fails, block the whole request:

  1. Quantum channel  - a quantum core in C_Q, a single contiguous slot honouring
                        continuity, contiguity, the guard band and the XT-avoided
                        mask.
  2. Companion channel - reuse the quantum route; a non-quantum core with
                        sufficient separation and a single slot.
  3. Data channel     - modulation/slot count from route distance; among the
                        candidate routes link-disjoint from the QC route, choose
                        the one minimising alpha*hops + beta*fragmentation; place
                        the slots on a non-quantum core under the XT-avoided mask.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib import request

from config import Config
from modulation import ModulationSelector
from spectrum import Allocation, SpectrumState
from topology import Path, Topology
from traffic import Request


@dataclass
class AllocationResult:
    success: bool
    allocations: List[Allocation] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    dc_route_idx: Optional[int] = None
    modulation: Optional[str] = None


# ---------------------------------------------------------------------------
# Quantum (route, core) feasibility - used for action masking and allocation
# ---------------------------------------------------------------------------
def qc_route_core_feasible(spectrum: SpectrumState, topo: Topology,
                           path: Path, q_core: int, cfg: Config) -> bool:
    """True if a quantum channel could be placed on (path, q_core): the core
    must be a quantum-dedicated core and a single slot must be placeable under
    continuity, contiguity, the guard band and the XT-avoided mask (Section 9)."""
    if q_core not in cfg.net.quantum_cores:
        return False
    return spectrum.can_place(path, q_core, 1)


# ---------------------------------------------------------------------------
# Core ordering helpers
# ---------------------------------------------------------------------------
def _companion_core_order(topo: Topology, cfg: Config, q_core: int) -> List[int]:
    """Non-quantum cores ordered to give the companion channel sufficient
    separation from the quantum core (prefer cores not adjacent to q_core)."""
    adj = set(topo.adjacent_cores(q_core))
    return sorted(cfg.net.non_quantum_cores, key=lambda c: (c in adj, c))


# ---------------------------------------------------------------------------
# Main allocation
# ---------------------------------------------------------------------------
def allocate(spectrum: SpectrumState, topo: Topology, modsel: ModulationSelector,
             cfg: Config, request: Request, candidate_routes: List[Path],
             route_idx: int, q_core: int,
             fit_qc: str, fit_cc: str, fit_dc: str) -> AllocationResult:

    if not candidate_routes or route_idx >= len(candidate_routes):
        return AllocationResult(False, blocked_reason="no_route")

    qc_route = candidate_routes[route_idx]
    committed: List[Allocation] = []

    # ---- 1. Quantum channel -------------------------------------------------
    if q_core not in cfg.net.quantum_cores:
        return AllocationResult(False, blocked_reason="invalid_quantum_core")
    start = spectrum.select_start(qc_route, q_core, 1, fit_qc)
    if start is None:
        return AllocationResult(False, blocked_reason="qc_spectrum")
    committed.append(spectrum.commit(qc_route, q_core, start, 1, "QC"))

    # ---- 2. Companion channel (same route, non-quantum core) ----------------
    cc_done = False
    for cc_core in _companion_core_order(topo, cfg, q_core):
        start = spectrum.select_start(qc_route, cc_core, 1, fit_cc)
        if start is not None:
            committed.append(spectrum.commit(qc_route, cc_core, start, 1, "CC"))
            cc_done = True
            break
    if not cc_done:
        spectrum.release_all(committed)
        return AllocationResult(False, blocked_reason="cc_spectrum")

    # ---- 3. Data channel (link-disjoint route, non-quantum core) ------------
    # Candidate DC routes: link-disjoint from the QC route (Section 3).
    dc_candidates: List[Tuple[int, Path, int]] = []  # (idx, route, slots)
    for idx, route in enumerate(candidate_routes):
        if not Topology.link_disjoint(qc_route, route):
            continue
        sel = modsel.select(topo.path_length(route), request.demand)
        if sel is None:
            continue                          # unreachable by any modulation
        _, _, slots = sel
        dc_candidates.append((idx, route, slots))

    if not dc_candidates:
        spectrum.release_all(committed)
        return AllocationResult(False, blocked_reason="dc_no_disjoint_route")

    # Choose the route minimising alpha*hops + beta*fragmentation.
    def route_cost(route: Path) -> float:
        hops = topo.path_hops(route)
        frag = _route_fragmentation(spectrum, topo, cfg, route)
        return cfg.alloc.alpha * hops + cfg.alloc.beta * frag

    dc_candidates.sort(key=lambda t: route_cost(t[1]))

    dc_done = False
    chosen_idx = None
    chosen_mod = None
    for idx, route, slots in dc_candidates:
        for dc_core in cfg.net.non_quantum_cores:
            start = spectrum.select_start(route, dc_core, slots, fit_dc)
            if start is not None:
                committed.append(spectrum.commit(route, dc_core, start, slots, "DC"))
                sel = modsel.select(topo.path_length(route), request.demand)
                chosen_mod = sel[1] if sel else None
                chosen_idx = idx
                dc_done = True
                break
        if dc_done:
            break

    if not dc_done:
        spectrum.release_all(committed)
        return AllocationResult(False, blocked_reason="dc_spectrum")

    return AllocationResult(True, allocations=committed,
                            dc_route_idx=chosen_idx, modulation=chosen_mod)


def _route_fragmentation(spectrum: SpectrumState, topo: Topology, cfg: Config,
                         route: Path) -> float:
    """Mean external fragmentation over the non-quantum cores of a route's
    links (the F(p) term of the data-channel cost)."""
    import numpy as np
    ratios = []
    for e in route:
        eid = topo.edge_index[e]
        for c in cfg.net.non_quantum_cores:
            free = ~spectrum.occupancy[eid, c, :]
            total_free = int(free.sum())
            if total_free == 0:
                ratios.append(1.0)
                continue
            largest = int(SpectrumState._run_lengths(free).max(initial=0))
            ratios.append(1.0 - largest / total_free)
    return float(np.mean(ratios)) if ratios else 0.0
