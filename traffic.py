"""
traffic.py
==========
Dynamic traffic + channel model for the QKD-enabled SS-EON.

Implements:
  - Section 3: each request is a triplet (QC, CC, DC).
        QC -> 1 slot, CC -> 1 slot, DC -> modulation/capacity dependent.
  - Section 4: dynamic requests with a uniformly random (src, dst) pair,
        data demand from capacity classes {50,100,...,400} Gbps,
        exponential holding time (mean H), Poisson arrivals (rate lambda);
        offered load rho = lambda * H Erlangs.
  - Section 7 (DC): modulation format from route distance -> DC slot count.

This mirrors the dynamic request loop in the user's main1.py (Poisson arrival
times, exponential holding times, random s-d pairs) and the distance->bitrate
slot derivation in functions.calculateRequiredSlots.

References (uploaded by the user):
  - implementation_v2.docx ....... spec (Sections 3, 4, 7).
  - main1.py ..................... user's dynamic request generation loop.
  - functions.py ................. user's BitRateAndDistances / required-slots.
"""

import numpy as np


# Section 4: data demand capacity classes (Gbps).
CAPACITY_CLASSES = [50, 100, 150, 200, 250, 300, 350, 400]

# Section 3: fixed single-slot quantum and companion channels.
QC_SLOTS = 1
CC_SLOTS = 1

# Slot width (GHz). Standard flexible-grid granularity.
SLOT_WIDTH_GHZ = 12.5

# Section 7: distance-dependent modulation format -> spectral efficiency
# (bits/symbol). Longer reach -> lower-order modulation -> more slots.
# Same distance-banding idea as functions.BitRateAndDistances, expressed as
# modulation efficiency. Distances are in km.
MODULATION_BY_DISTANCE = [
    (0,    500,   4),   # 16-QAM   : 4 bits/symbol
    (500,  1000,  3),   # 8-QAM    : 3 bits/symbol
    (1000, 2000,  2),   # QPSK     : 2 bits/symbol
    (2000, 1e9,   1),   # BPSK     : 1 bit/symbol
]


def modulation_efficiency(distance_km):
    """Bits/symbol for a route of the given length (Section 7)."""
    for lo, hi, bits in MODULATION_BY_DISTANCE:
        if lo <= distance_km < hi:
            return bits
    return MODULATION_BY_DISTANCE[-1][2]


def dc_slot_count(demand_gbps, distance_km):
    """DC slot count from capacity class + modulation format (Section 7).

    slots = ceil( demand / (bits_per_symbol * slot_width) ).
    The 1-slot guard band (Section 5) is applied at allocation time in
    mcf_resources, not added here.
    """
    bits = modulation_efficiency(distance_km)
    return int(np.ceil(demand_gbps / (bits * SLOT_WIDTH_GHZ)))


class TrafficGenerator:
    """Poisson arrivals + exponential holding times of (QC, CC, DC) triplets."""

    def __init__(self, nodes, arrival_rate, holding_mean,
                 capacity_classes=None, seed=None):
        self.nodes = list(nodes)
        self.arrival_rate = arrival_rate          # lambda
        self.holding_mean = holding_mean          # H
        self.capacity_classes = capacity_classes or CAPACITY_CLASSES
        self.rng = np.random.default_rng(seed)

    @property
    def offered_load(self):
        """rho = lambda * H Erlangs (Section 4)."""
        return self.arrival_rate * self.holding_mean

    def generate(self, num_requests):
        """Yield request dicts in arrival order.

        Inter-arrival times ~ Exponential(1/lambda) (a Poisson process);
        holding times ~ Exponential(mean = H).
        """
        t = 0.0
        for rid in range(num_requests):
            t += self.rng.exponential(1.0 / self.arrival_rate)
            holding = self.rng.exponential(self.holding_mean)
            src, dst = self.rng.choice(self.nodes, size=2, replace=False)
            demand = int(self.rng.choice(self.capacity_classes))
            yield {
                "id": rid,
                "src": int(src),
                "dst": int(dst),
                "arrival": t,
                "holding": holding,
                "demand": demand,     # DC data demand (Gbps)
                "qc_slots": QC_SLOTS,
                "cc_slots": CC_SLOTS,
            }
