"""
Modulation / slot-count derivation for the data channel (Section 7).

"Select the modulation format from route distance and derive the DC slot count."

We use distance-adaptive modulation: pick the highest-order format whose optical
reach still covers the path length, then compute the number of frequency slots
following DeepRMSA Eq.(1) generalised to the six PM formats of Xav-RQCD:

        n = ceil( demand / (m * C_BPSK_per_slot) )

where m is the modulation's bits-per-symbol. QC and CC always need 1 slot
(Section 3); only the DC slot count depends on modulation.
"""

import math
from typing import Optional, Tuple

from config import ModulationConfig


class ModulationSelector:
    def __init__(self, cfg: ModulationConfig):
        self.cfg = cfg
        # Order formats from highest reach (lowest order) to lowest reach.
        order = sorted(range(len(cfg.names)), key=lambda i: cfg.reach_km[i],
                       reverse=True)
        self._order = order

    def select(self, path_length_km: float, demand_gbps: int
               ) -> Optional[Tuple[int, str, int]]:
        """Return (modulation_index, name, slots) for the most spectrally
        efficient format that still reaches `path_length_km`, or None if even
        the most robust format cannot reach.
        """
        best = None
        for i in self._order:
            if self.cfg.reach_km[i] >= path_length_km:
                # Among reachable formats, prefer the highest bits-per-symbol
                # (fewest slots). Iterating reach-descending and keeping the
                # last reachable gives the highest order that still reaches.
                best = i
        if best is None:
            return None
        m = self.cfg.bits_per_symbol[best]
        slots = math.ceil(demand_gbps / (m * self.cfg.c_bpsk_per_slot))
        return best, self.cfg.names[best], int(slots)
