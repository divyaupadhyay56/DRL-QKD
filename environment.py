import itertools
import numpy as np

import mcf_resources as mr
from traffic import dc_slot_count


FIT_STRATEGIES = ["first", "best", "random"]

ALPHA_HOPS = 1.0
BETA_FRAG = 1.0

FEATURES_PER_PATH_CORE = 5


class QKDEnvironment:

    def __init__(self, network, traffic_gen,
                 k_paths=5,
                 total_requests=100000,num_training_slots=250):

        self.net = network
        self.traffic = traffic_gen
        self.k = k_paths
        self.total_requests = total_requests
        self.num_training_slots = num_training_slots
        # self.max_training_slots = 100

        self.slot_table = mr.build_slot_table(
            self.net.num_links,
            self.net.cores_per_link,
            self.net.slots_per_core
        )

        self.active = {}

        self.action_table = list(
            itertools.product(
                range(self.k),
                range(len(self.net.quantum_cores)),
                range(len(FIT_STRATEGIES)),
                range(len(FIT_STRATEGIES)),
                range(len(FIT_STRATEGIES))
            )
        )

        self.action_size = len(self.action_table)

        n = len(self.net.nodes)

        self.state_size = (
            self.k *
            self.net.cores_per_link *
            FEATURES_PER_PATH_CORE
            + 2 * n
            + 3
        )

        self._request_iter = None
        self._reset_iter()

    # -------------------------------------------------

    def _reset_iter(self):
        self._request_iter = self.traffic.generate(
            self.total_requests
        )

    # -------------------------------------------------

    def _candidate_routes(self, request):
        return self.net.k_shortest_paths(
            request["src"],
            request["dst"],
            self.k
        )

    # -------------------------------------------------

    def _dc_slots_for_route(self, request, link_ids):
        return dc_slot_count(
            request["demand"],
            self.net.path_distance(link_ids)
        )

    # =================================================
    # STATE
    # =================================================

    def build_state(self, request, routes):

        n = len(self.net.nodes)

        S = self.net.slots_per_core

        feats = []

        for r in range(self.k):

            if r < len(routes):
                link_ids = routes[r]
                dc_slots = self._dc_slots_for_route(
                    request,
                    link_ids
                )
            else:
                link_ids = None
                dc_slots = 0

            for core in range(
                    self.net.cores_per_link):

                if link_ids is None:
                    feats.extend([0, 0, 0, 0, 0])
                    continue

                occ = mr.feasible_occupancy(
                    self.slot_table,
                    link_ids,
                    core,
                    self.net.core_adjacency,
                    self.num_training_slots

                )

                blocks = mr.free_blocks(occ)

                if blocks:

                    first_start = blocks[0][0]

                    first_size = len(blocks[0])

                    total_avail = sum(
                        len(b)
                        for b in blocks
                    )

                    mean_size = (
                        total_avail /
                        len(blocks)
                    )

                else:

                    first_start = 0
                    first_size = 0
                    total_avail = 0
                    mean_size = 0

                feats.extend([
                    first_start / S,
                    first_size / S,
                    total_avail / S,
                    mean_size / S,
                    dc_slots / S
                ])

        src_oh = [0] * n
        dst_oh = [0] * n

        src_oh[
            self.net.nodes.index(
                request["src"])
        ] = 1

        dst_oh[
            self.net.nodes.index(
                request["dst"])
        ] = 1

        descriptor = src_oh + dst_oh + [

            request["qc_slots"],

            request["cc_slots"],

            self._dc_slots_for_route(
                request,
                routes[0]
            ) / S if routes else 0
        ]

        return np.asarray(
            feats + descriptor,
            dtype=np.float32
        )

    # =================================================
    # ACTION MASK
    # =================================================

    def action_mask(self, request, routes):

        mask = np.zeros(
            self.action_size,
            dtype=np.float32
        )

        for a, (ri, qci, _, _, _) in enumerate(
                self.action_table):

            if ri >= len(routes):
                continue

            qc_route = routes[ri]

            qcore = self.net.quantum_cores[qci]

            occ = mr.feasible_occupancy(
                self.slot_table,
                qc_route,
                qcore,
                self.net.core_adjacency,
                self.num_training_slots
            )

            if not mr._candidate_blocks(
                    occ,
                    request["qc_slots"]):
                continue

            disjoint = [
                r for r in routes
                if self.net.link_disjoint(
                    r,
                    qc_route
                )
            ]

            if len(disjoint) == 0:
                continue

            mask[a] = 1.0

        return mask

    # =================================================
    # FRAGMENTATION
    # =================================================

    def _route_fragmentation(self, link_ids):

        ratios = []

        for core in self.net.data_cores:

            occ = mr.route_core_occupancy(
                self.slot_table,
                link_ids,
                core
            )

            blocks = mr.free_blocks(occ)

            total_free = sum(
                len(b)
                for b in blocks
            )

            if total_free == 0:
                ratios.append(0.0)

            else:
                ratios.append(
                    1.0 -
                    max(len(b) for b in blocks)
                    / total_free
                )

        return (
            sum(ratios) / len(ratios)
            if ratios else 0.0
        )

    # =================================================

    def _allocate_channel(
            self,
            link_ids,
            cores,
            required,
            strategy):

        for core in cores:

            occ = mr.feasible_occupancy(
                self.slot_table,
                link_ids,
                core,
                self.net.core_adjacency,
                self.num_training_slots
            )

            start = mr.select_start(
                occ,
                required,
                strategy
            )

            if start is not None:

                idx = mr.occupy(
                    self.slot_table,
                    link_ids,
                    core,
                    start,
                    required
                )

                return core, idx

        return None, None

    # =================================================
    # STEP
    # =================================================

    def step(self, request, routes, action):

        ri, qci, qc_fit, cc_fit, dc_fit = \
            self.action_table[action]

        info = {
            "served": False,
            "reason": None
        }

        if ri >= len(routes):
            return -1.0, True, info

        qc_route = routes[ri]

        placed = []

        # ---------------- QC ----------------

        qcore = self.net.quantum_cores[qci]

        core, idx = self._allocate_channel(
            qc_route,
            [qcore],
            request["qc_slots"],
            FIT_STRATEGIES[qc_fit]
        )

        if core is None:
            info["reason"] = "qc"
            return -1.0, True, info

        placed.append(
            (qc_route, core, idx)
        )

        # ---------------- CC ----------------

        cc_cores = list(
            self.net.data_cores
        )

        core, idx = self._allocate_channel(
            qc_route,
            cc_cores,
            request["cc_slots"],
            FIT_STRATEGIES[cc_fit]
        )

        if core is None:

            self._rollback(placed)

            info["reason"] = "cc"

            return -1.0, True, info

        placed.append(
            (qc_route, core, idx)
        )

        # ---------------- DC ----------------

        disjoint = [

            r for r in routes

            if self.net.link_disjoint(
                r,
                qc_route
            )
        ]

        if len(disjoint) == 0:

            self._rollback(placed)

            info["reason"] = "disjoint"

            return -1.0, True, info

        dc_route = min(
            disjoint,

            key=lambda r:

            ALPHA_HOPS * len(r)

            + BETA_FRAG *
            self._route_fragmentation(r)
        )

        dc_slots = self._dc_slots_for_route(
            request,
            dc_route
        )

        core, idx = self._allocate_channel(
            dc_route,
            list(self.net.data_cores),
            dc_slots,
            FIT_STRATEGIES[dc_fit]
        )

        if core is None:

            self._rollback(placed)

            info["reason"] = "dc"

            return -1.0, True, info

        placed.append(
            (dc_route, core, idx)
        )

        self.active[
            request["id"]
        ] = {

            "release_time":
            request["arrival"]
            + request["holding"],

            "placements":
            placed
        }

        info["served"] = True
        info["reason"] = "success"

        return 1.0, False, info

    # =================================================

    def _rollback(self, placed):

        for link_ids, core, idx in placed:

            mr.release(
                self.slot_table,
                link_ids,
                core,
                idx
            )

    # =================================================

    def release_expired(self, current_time):

        for rid in list(self.active):

            if current_time >= \
                    self.active[rid]["release_time"]:

                for link_ids, core, idx in \
                        self.active[rid]["placements"]:

                    mr.release(
                        self.slot_table,
                        link_ids,
                        core,
                        idx
                    )

                del self.active[rid]

    # =================================================

    def reset(self):

        mr.reset_slot_table(
            self.slot_table
        )

        self.active.clear()

        self._reset_iter()




