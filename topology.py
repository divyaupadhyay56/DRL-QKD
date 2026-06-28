import networkx as nx


# ---------------------------------------------------------------------------
# Section 2: Concrete MCF setup (exact values from implementation_v2)
# ---------------------------------------------------------------------------
CORES_PER_LINK = 5          # 5 cores per link
SLOTS_PER_CORE = 320        # 320 frequency slots per core
QUANTUM_CORES = (0, 1)      # set C_Q: 2 quantum-dedicated cores
DATA_CORES = (2, 3, 4)      # remaining, non-quantum cores (CC and DC)


# ---------------------------------------------------------------------------
# Tokyo12 MAN network: 12 nodes, bidirectional fiber links with length (km).
# (Standard Japanese 12-node backbone link set.)
# ---------------------------------------------------------------------------
TOKYO12_NODES = list(range(12))

TOKYO12_LINKS = [
    (0, 1, 593), (0, 2, 1256),
    (1, 2, 351), (1, 3, 47),
    (2, 4, 250), (2, 5, 252),
    (3, 4, 252), (3, 6, 314),
    (4, 7, 224), (5, 7, 207),
    (5, 8, 350), (6, 7, 268),
    (6, 9, 191), (7, 8, 263),
    (7, 10, 311), (8, 11, 247),
    (9, 10, 260), (10, 11, 296),
]


def inter_core_adjacency(num_cores=CORES_PER_LINK):
    """Section 2/6: inter-core adjacency matrix.

    Returns a num_cores x num_cores 0/1 matrix where entry (i, j) == 1 means
    cores i and j are physically neighbouring on the same link and therefore
    constrained by the XT-avoided rule (Section 6). A standard 5-core layout
    is one central core (4) adjacent to the four outer cores (0..3) arranged in
    a ring, so each outer core neighbours the centre and its two ring
    neighbours. Edit this matrix to match a different physical core layout.

    Scope hook: extend with an inter-mode dimension here when moving beyond the
    single-mode-per-core assumption.
    """
    adj = [[0] * num_cores for _ in range(num_cores)]
    if num_cores == 5:
        neighbours = {
            0: [1, 3, 4],
            1: [0, 2, 4],
            2: [1, 3, 4],
            3: [0, 2, 4],
            4: [0, 1, 2, 3],
        }
        for c, nbrs in neighbours.items():
            for n in nbrs:
                adj[c][n] = 1
    else:
        # Fallback: linear nearest-neighbour adjacency.
        for c in range(num_cores - 1):
            adj[c][c + 1] = 1
            adj[c + 1][c] = 1
    return adj


class MCFNetwork:
    """Multi-core-fiber network graph.

    Mirrors the role of `networkGraph` in the user's graph_functions.py
    (node list, indexed edge list, k-shortest paths, path distances) but adds
    the MCF dimension: every directed link owns CORES_PER_LINK cores, and the
    spectrum state is held per (link, core).
    """

    def __init__(self, nodes=None, links=None,
                 cores_per_link=CORES_PER_LINK,
                 slots_per_core=SLOTS_PER_CORE,
                 quantum_cores=QUANTUM_CORES,
                 data_cores=DATA_CORES):
        self.cores_per_link = cores_per_link
        self.slots_per_core = slots_per_core
        self.quantum_cores = tuple(quantum_cores)
        self.data_cores = tuple(data_cores)
        self.core_adjacency = inter_core_adjacency(cores_per_link)

        nodes = TOKYO12_NODES if nodes is None else nodes
        links = TOKYO12_LINKS if links is None else links

        # Directed graph (Section 2: model as a directed graph).
        self.g = nx.DiGraph()
        self.g.add_nodes_from(nodes)
        for u, v, dist in links:
            self.g.add_edge(u, v, weight=dist)
            self.g.add_edge(v, u, weight=dist)

        self.nodes = list(self.g.nodes)
        self.edges = list(self.g.edges)                  # indexed directed links
        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        self.num_links = len(self.edges)

    # --- routing (same idea as graph_functions.findPaths / findDistances) ----
    def k_shortest_paths(self, source, destination, k):
        """K candidate routes as ordered lists of directed-link indices.

        Section 7: "Compute K candidate routes (K-shortest paths) ... (k upto 8)."
        """
        paths = nx.shortest_simple_paths(self.g, source, destination,
                                         weight="weight")
        ksp = []
        for number, node_path in enumerate(paths):
            link_ids = [self.edge_index[(node_path[i], node_path[i + 1])]
                        for i in range(len(node_path) - 1)]
            ksp.append(link_ids)
            if number == k - 1:
                break
        return ksp

    def path_distance(self, link_ids):
        """Total length (km) of a route given as link indices."""
        return sum(self.g.edges[self.edges[lid]]["weight"] for lid in link_ids)

    @staticmethod
    def link_disjoint(path_a, path_b):
        """Section 3/7: DC must be link-disjoint from the QC/CC route."""
        return len(set(path_a) & set(path_b)) == 0
