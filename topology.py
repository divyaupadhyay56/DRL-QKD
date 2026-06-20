"""
Network model (Section 2 of implementation_v2).

- The optical network is a *directed* graph: nodes are optical cross-connects,
  links are multi-core-fiber connections.
- An inter-core adjacency matrix describes which cores are physically
  neighbouring (used for the XT-avoided crosstalk relationships). The matrix is
  built so it can later be extended with an inter-mode dimension.
- K-shortest paths are computed per source-destination pair (Section 7).
"""

from functools import lru_cache
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

Edge = Tuple[int, int]          # a directed link (u, v)
Path = List[Edge]               # an ordered list of directed links


# ---------------------------------------------------------------------------
# Topology definitions
# ---------------------------------------------------------------------------
# 6-node network used for MILP-optimal validation (Xav-RQCD Fig.3(a)):
# eight bidirectional links with lengths {500,700,600,400,450,500,450,400} km.
_SIXNODE_LINKS = [
    (0, 1, 500.0), (1, 2, 700.0), (2, 3, 600.0), (3, 4, 400.0),
    (4, 5, 450.0), (5, 0, 500.0), (0, 2, 450.0), (3, 5, 400.0),
]

# Tokyo12 metropolitan-area network (the topology in networks.pdf, diagram (b)).
# Read directly from the figure: 12 nodes, 21 bidirectional links, lengths in km.
# The figure labels nodes 1..12; here node index = figure_label - 1.
# (u, v, km) with u, v as zero-based indices:
_TOKYO12_LINKS = [
    (0, 2, 6.2),   (2, 4, 6.7),   (2, 3, 7.1),   (0, 3, 7.5),   (0, 1, 10.5),
    (4, 9, 10.1),  (9, 11, 10.5), (4, 5, 1.9),   (5, 11, 11.5), (5, 10, 7.0),
    (10, 11, 7.6), (3, 5, 5.0),   (3, 6, 7.1),   (5, 6, 5.0),   (1, 3, 8.0),
    (1, 6, 10.3),  (6, 10, 7.5),  (1, 7, 9.1),   (1, 8, 12.4),  (6, 7, 6.7),
    (7, 8, 6.0),
]

_TOPOLOGIES = {
    "sixnode": _SIXNODE_LINKS,
    "tokyo12": _TOKYO12_LINKS,
}


class Topology:
    """Directed MCF graph plus the inter-core adjacency matrix."""

    def __init__(self, name: str, num_cores: int, adjacency_layout: str = "ring"):
        if name not in _TOPOLOGIES:
            raise ValueError(f"unknown topology '{name}'")
        self.name = name
        self.num_cores = num_cores

        # Build a directed graph (each undirected fiber becomes two links).
        self.graph = nx.DiGraph()
        for u, v, km in _TOPOLOGIES[name]:
            self.graph.add_edge(u, v, length=km)
            self.graph.add_edge(v, u, length=km)

        self.nodes = sorted(self.graph.nodes())
        self.num_nodes = len(self.nodes)

        # Stable indexing of directed links.
        self.edges: List[Edge] = sorted(self.graph.edges())
        self.edge_index: Dict[Edge, int] = {e: i for i, e in enumerate(self.edges)}
        self.num_edges = len(self.edges)

        self.core_adjacency = self._build_core_adjacency(adjacency_layout)

    # ---- inter-core adjacency ------------------------------------------------
    def _build_core_adjacency(self, layout: str) -> np.ndarray:
        """Binary matrix: A[i, j] = 1 if cores i and j are physical neighbours.

        For a 5-core fiber the layout is a ring (Xav-RQCD Fig.2(b)):
        core i neighbours i-1 and i+1 (mod num_cores). The matrix is the hook
        for an eventual inter-mode extension.
        """
        n = self.num_cores
        A = np.zeros((n, n), dtype=np.int8)
        if layout == "ring":
            for i in range(n):
                A[i, (i + 1) % n] = 1
                A[i, (i - 1) % n] = 1
        elif layout == "linear":
            for i in range(n - 1):
                A[i, i + 1] = 1
                A[i + 1, i] = 1
        else:
            raise ValueError(f"unknown adjacency layout '{layout}'")
        return A

    def adjacent_cores(self, core: int) -> List[int]:
        return [c for c in range(self.num_cores) if self.core_adjacency[core, c]]

    # ---- routing -------------------------------------------------------------
    @lru_cache(maxsize=None)
    def k_shortest_paths(self, src: int, dst: int, k: int) -> Tuple[Path, ...]:
        """Return up to k length-shortest simple paths as ordered edge lists."""
        if src == dst:
            return tuple()
        gen = nx.shortest_simple_paths(self.graph, src, dst, weight="length")
        paths: List[Path] = []
        for node_path in gen:
            edge_path = list(zip(node_path[:-1], node_path[1:]))
            paths.append(edge_path)
            if len(paths) >= k:
                break
        return tuple(paths)

    def path_length(self, path: Path) -> float:
        return sum(self.graph[u][v]["length"] for (u, v) in path)

    def path_hops(self, path: Path) -> int:
        return len(path)

    @staticmethod
    def link_disjoint(path_a: Path, path_b: Path) -> bool:
        """True if the two paths share no link (Section 3: DC must be
        link-disjoint from the QC/CC route)."""
        sa = set(path_a)
        return all(e not in sa for e in path_b)
