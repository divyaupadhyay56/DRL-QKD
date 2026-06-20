"""
Central configuration for the DRL-based joint resource allocation in a
QKD-enabled SS-EON over multi-core fiber.

Every value here maps directly to a statement in implementation_v2. The only
constants imported as *reference* from the papers are the modulation reach table
and the BPSK per-slot capacity (DeepRMSA Eq.(1)); they are kept here so they can
be edited in one place and add no behaviour beyond what the doc asks for.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class NetworkConfig:
    # ---- Section 2: Network Model -------------------------------------------
    # "Network = Tokyo12 MAN network" (the topology in networks.pdf, diagram b).
    topology: str = "tokyo12"             # or "sixnode" for MILP-optimal validation
    num_cores: int = 5
    slots_per_core: int = 320

    # A subset of cores is reserved exclusively for quantum channels (the set C_Q).
    # Xav-RQCD dedicates the first two cores (c1, c2) to quantum traffic.
    quantum_cores: List[int] = field(default_factory=lambda: [0, 1])

    def __post_init__(self):
        self.non_quantum_cores = [c for c in range(self.num_cores)
                                  if c not in self.quantum_cores]


@dataclass
class RequestConfig:
    # ---- Section 4: Dynamic Traffic Generation ------------------------------
    # Data demand drawn from the capacity classes {50, 100, ..., 400} Gbps.
    capacity_classes: List[int] = field(
        default_factory=lambda: [50, 100, 150, 200, 250, 300, 350, 400])

    # Poisson arrivals (rate lambda); exponential holding time (mean H).
    # Offered load rho = lambda * H Erlangs.
    arrival_rate: float = 12.0            # lambda (requests / time unit)
    mean_holding_time: float = 6.0       # H  -> rho = 240 Erlangs by default


@dataclass
class ModulationConfig:
    # ---- Section 7: Data channel -> "select the modulation format from route
    #      distance and derive the DC slot count" ------------------------------
    # Distance-adaptive modulation, generalising DeepRMSA Eq.(1)
    #   n = ceil( b / (m * C_BPSK_grid) )
    # to the six PM formats listed by Xav-RQCD.
    #
    # names / bits-per-symbol m / optical reach in km.
    # Reach values follow standard distance-adaptive flexgrid practice; adjust
    # to your link budget if the JPN reference specifies different d_theta.
    names:  List[str]   = field(default_factory=lambda: [
        "PM-BPSK", "PM-QPSK", "PM-8QAM", "PM-16QAM", "PM-32QAM", "PM-64QAM"])
    bits_per_symbol: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    reach_km: List[float] = field(default_factory=lambda: [
        9000.0, 4500.0, 2000.0, 1000.0, 500.0, 250.0])

    c_bpsk_per_slot: float = 12.5         # Gb/s carried by one FS of BPSK


@dataclass
class XTConfig:
    # ---- Section 6: Crosstalk Modelling (XT-Avoided) ------------------------
    # Inter-core crosstalk is a *binary feasibility mask*: forbid the same slot
    # on adjacent cores of the same link for different connections.
    # The 5-core layout in Xav-RQCD Fig.2(b) is a ring: c1-c2-c3-c4-c5-c1,
    # so core i is adjacent to its two cycle neighbours.
    adjacency_layout: str = "ring"


@dataclass
class AllocationConfig:
    # ---- Section 5 & 7 ------------------------------------------------------
    k_paths: int = 8                      # K candidate routes, k up to 8
    guard_band: int = 1                   # 1 slot between adjacent allocations

    # Section 7 data-channel objective: minimise alpha*(hops) + beta*(frag).
    # Xav-RQCD uses alpha = 0.6, beta = 0.4.
    alpha: float = 0.6
    beta: float = 0.4


@dataclass
class DRLConfig:
    # ---- Section 9-10 -------------------------------------------------------
    fit_strategies: List[str] = field(default_factory=lambda: ["first", "best", "random"])
    episode_length: int = 100             # requests serviced per episode

    # PPO hyper-parameters (Section 10: "Algorithm: PPO ... clipped objective").
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])

    # Offline-then-online schedule (Section 10).
    offline_requests: int = 50_000        # pre-training on the simulator
    online_requests: int = 20_000         # online fine-tuning
    rollout_requests: int = 2_048         # requests collected per PPO update


@dataclass
class Config:
    net: NetworkConfig = field(default_factory=NetworkConfig)
    req: RequestConfig = field(default_factory=RequestConfig)
    mod: ModulationConfig = field(default_factory=ModulationConfig)
    xt: XTConfig = field(default_factory=XTConfig)
    alloc: AllocationConfig = field(default_factory=AllocationConfig)
    drl: DRLConfig = field(default_factory=DRLConfig)
    seed: int = 0


DEFAULT_CONFIG = Config()
