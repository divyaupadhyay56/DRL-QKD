"""
Dynamic traffic generation (Section 4 of implementation_v2).

Each request is a triplet (QC, CC, DC) with:
  - a uniformly random source/destination pair,
  - a data demand drawn from the capacity classes {50,...,400} Gbps,
  - an exponentially distributed holding time (mean H),
  - arrivals following a Poisson process of rate lambda.

The generator is event-driven and supports a mid-run change of the traffic
distribution, used by the robustness evaluation (Section 10).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Request:
    rid: int
    src: int
    dst: int
    demand: int                 # data-channel demand in Gbps
    arrival_time: float
    holding_time: float

    @property
    def departure_time(self) -> float:
        return self.arrival_time + self.holding_time


class TrafficGenerator:
    """Produces a stream of requests with exponential inter-arrival / holding."""

    def __init__(self, num_nodes: int, capacity_classes: List[int],
                 arrival_rate: float, mean_holding_time: float, seed: int = 0):
        self.num_nodes = num_nodes
        self.capacity_classes = list(capacity_classes)
        self.arrival_rate = arrival_rate
        self.mean_holding_time = mean_holding_time
        self.rng = np.random.default_rng(seed)

        self._clock = 0.0
        self._next_id = 0
        # Optional non-uniform source/destination weights for the robustness test.
        self._node_weights: Optional[np.ndarray] = None

    # ---- distribution control -----------------------------------------------
    def set_uniform(self):
        """Uniform traffic distribution (the default)."""
        self._node_weights = None

    def set_nonuniform(self, concentration: float = 4.0):
        """Skew the src/dst distribution toward a subset of hot nodes.

        This implements the mid-run traffic shift used to check that the policy
        is not overfit to a single traffic pattern (Section 10)."""
        w = self.rng.random(self.num_nodes) ** concentration
        self._node_weights = w / w.sum()

    def set_load(self, arrival_rate: float = None, mean_holding_time: float = None):
        if arrival_rate is not None:
            self.arrival_rate = arrival_rate
        if mean_holding_time is not None:
            self.mean_holding_time = mean_holding_time

    # ---- sampling ------------------------------------------------------------
    def _sample_pair(self):
        if self._node_weights is None:
            src, dst = self.rng.choice(self.num_nodes, size=2, replace=False)
        else:
            src, dst = self.rng.choice(self.num_nodes, size=2, replace=False,
                                       p=self._node_weights)
        return int(src), int(dst)

    def next_request(self) -> Request:
        # Poisson process: exponential inter-arrival time.
        self._clock += self.rng.exponential(1.0 / self.arrival_rate)
        src, dst = self._sample_pair()
        demand = int(self.rng.choice(self.capacity_classes))
        holding = float(self.rng.exponential(self.mean_holding_time))
        req = Request(self._next_id, src, dst, demand, self._clock, holding)
        self._next_id += 1
        return req

    @property
    def offered_load(self) -> float:
        """rho = lambda * H Erlangs."""
        return self.arrival_rate * self.mean_holding_time

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._clock = 0.0
        self._next_id = 0
