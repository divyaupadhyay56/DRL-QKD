import itertools
import numpy as np

import mcf_resources as mr
from traffic import dc_slot_count


# ==========================================================
# Block-selection action design (DeepRMSA J-block reference)
# ==========================================================
# Action = (route index,
#           quantum-core index,
#           QC block choice in [0, J_BLOCKS),
#           CC block choice in [0, J_BLOCKS),
#           DC block choice in [0, J_BLOCKS))
#
# Feasible blocks per channel are ordered by spectral position
# (start slot, then core), so block index j always refers to the
# j-th earliest feasible block. The SAME candidate lists are used
# to build the state, the action mask and the allocation, so what
# the agent sees is exactly what its action selects. Choosing a
# CC/DC block also chooses its core (the core is part of the
# candidate), so core selection belongs to the agent, not to the
# environment.
J_BLOCKS = 3   # default; override via QKDEnvironment(j_blocks=...)

# Section 7: DC-route heuristic weights
# (minimise alpha * hops + beta * fragmentation).
ALPHA_HOPS = 1.0
BETA_FRAG = 1.0

# Global spectrum descriptors: fragmentation + utilisation.
GLOBAL_FEATURES = 2


class QKDEnvironment:

    def __init__(self, network, traffic_gen,
                 k_paths=5,
                 total_requests=100000, num_training_slots=250,
                 j_blocks=J_BLOCKS):

        self.j = j_blocks

        self.net = network
        self.traffic = traffic_gen
        self.k = k_paths
        self.total_requests = total_requests
        self.num_training_slots = num_training_slots

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
                range(self.j),   # QC block choice
                range(self.j),   # CC block choice
                range(self.j)    # DC block choice
            )
        )

        self.action_size = len(self.action_table)

        # ---- channel-aware state layout, per candidate route ----
        # QC, per quantum core : J x (start, size) + block count
        # CC                   : J x (start, size, core) + count
        # DC                   : J x (start, size, core) + count
        #                        + dc route exists flag
        #                        + dc slots required
        nq = len(self.net.quantum_cores)

        self.features_per_route = (
            nq * (2 * self.j + 1)
            + (3 * self.j + 1)
            + (3 * self.j + 3)
        )

        n = len(self.net.nodes)

        self.state_size = (
            self.k * self.features_per_route
            + GLOBAL_FEATURES
            + 2 * n
            + 3
        )

        # per-request shared candidate-block cache
        self._cache = None

        self._request_iter = None
        self._reset_iter()

    # -------------------------------------------------

    def _reset_iter(self):
        self._request_iter = self.traffic.generate(
            self.total_requests
        )

    # -------------------------------------------------
    # Flow: Find K Candidate Routes (K-Shortest)
    # -------------------------------------------------

    def _candidate_routes(self, request):
        return self.net.k_shortest_paths(
            request["src"],
            request["dst"],
            self.k
        )

    # -------------------------------------------------
    # Flow: Calculate Required Slots using Modulation
    # -------------------------------------------------

    def _dc_slots_for_route(self, request, link_ids):
        return dc_slot_count(
            request["demand"],
            self.net.path_distance(link_ids)
        )

    # =================================================
    # Flow: Find DC Route (Link-Disjoint from QC/CC)
    # Section 7: min alpha * hops + beta * fragmentation.
    # Route fragmentation is memoised per request.
    # =================================================

    def _dc_route(self, routes, qc_route, frag_memo):

        disjoint = [
            r for r in routes
            if self.net.link_disjoint(r, qc_route)
        ]

        if not disjoint:
            return None

        def cost(r):
            key = tuple(r)
            if key not in frag_memo:
                frag_memo[key] = \
                    self._route_fragmentation(r)
            return (ALPHA_HOPS * len(r)
                    + BETA_FRAG * frag_memo[key])

        return min(disjoint, key=cost)

    # =================================================
    # Flow: Generate ALL Feasible Free Blocks
    # (infeasible blocks, size < required + guard band,
    # are removed by mcf_resources.candidate_blocks)
    # =================================================

    def _channel_blocks(self, link_ids, cores, required):
        """Feasible free blocks for one channel as
        (core, start, size), sorted by spectral position."""

        blocks = []

        for core in cores:

            occ = mr.feasible_occupancy(
                self.slot_table,
                link_ids,
                core,
                self.net.core_adjacency,
                self.num_training_slots
            )

            for b in mr.candidate_blocks(occ, required):
                blocks.append(
                    (core, b[0], len(b))
                )

        blocks.sort(key=lambda x: (x[1], x[0]))

        # Only the first J blocks are ever accessible to the
        # agent, so anything beyond them is dropped here and
        # never stored, featurised or masked.
        return blocks[:self.j]

    # =================================================
    # SHARED CANDIDATE-BLOCK CACHE
    # Built once per request; reused by build_state,
    # action_mask and step, so candidate blocks are
    # never generated twice for the same request.
    # =================================================

    def _ensure_cache(self, request, routes):

        # Cache is valid only for this exact request AND the
        # spectrum state it was built from. It is invalidated
        # whenever the slot table mutates (allocation in step,
        # releases in release_expired, reset), so a stale view
        # can never leak into the state, mask or allocation.
        key = (request["id"], request["arrival"])

        if (self._cache is not None
                and self._cache["key"] == key):
            return self._cache

        cache = {
            "key": key,
            "qc": {},        # (ri, qci) -> feasible QC blocks
            "cc": {},        # ri -> feasible CC blocks
            "dc": {},        # ri -> feasible DC blocks
            "dc_route": {},  # ri -> DC route (or None)
            "dc_slots": {},  # ri -> DC slots required
            
        }

        frag_memo = {}

        data_cores = list(self.net.data_cores)

        for ri in range(min(self.k, len(routes))):

            qc_route = routes[ri]


            # route combination check (Flow: valid route?)
            dc_route = self._dc_route(
                routes, qc_route, frag_memo
            )

            cache["dc_route"][ri] = dc_route

            # QC feasible blocks, per quantum core
            for qci, qcore in enumerate(
                    self.net.quantum_cores):

                cache["qc"][(ri, qci)] = \
                    self._channel_blocks(
                        qc_route,
                        [qcore],
                        request["qc_slots"]
                    )

            # CC feasible blocks (same route, data cores)
            cache["cc"][ri] = self._channel_blocks(
                qc_route,
                data_cores,
                request["cc_slots"]
            )

            # DC feasible blocks (disjoint route)
            if dc_route is None:
                cache["dc"][ri] = []
                cache["dc_slots"][ri] = 0
            else:
                ds = self._dc_slots_for_route(
                    request, dc_route
                )
                cache["dc_slots"][ri] = ds
                cache["dc"][ri] = self._channel_blocks(
                    dc_route,
                    data_cores,
                    ds
                )

        # global metrics computed once per request
        cache["frag"] = mr.external_fragmentation(
            self.slot_table
        )
        cache["util"] = mr.spectrum_utilisation(
            self.slot_table
        )

        self._cache = cache

        return cache

    # =================================================
    # STATE
    # Channel-aware: the j-th (start, size[, core])
    # entry for QC / CC / DC of route ri is EXACTLY the
    # block that action indices (jq, jc, jd) select.
    # =================================================

    def build_state(self, request, routes):

        cache = self._ensure_cache(request, routes)

        n = len(self.net.nodes)
        S = self.net.slots_per_core
        C = self.net.cores_per_link

        feats = []

        for ri in range(self.k):

            if ri >= len(routes):
                feats.extend(
                    [0.0] * self.features_per_route
                )
                continue

            # ---- QC blocks, per quantum core ----
            for qci in range(
                    len(self.net.quantum_cores)):

                qlist = cache["qc"][(ri, qci)]

                for j in range(self.j):
                    if j < len(qlist):
                        _, start, size = qlist[j]
                        feats.extend(
                            [start / S, size / S]
                        )
                    else:
                        feats.extend([0.0, 0.0])

                feats.append(len(qlist) / S)

            # ---- CC blocks ----
            clist = cache["cc"][ri]

            for j in range(self.j):
                if j < len(clist):
                    core, start, size = clist[j]
                    feats.extend([
                        start / S,
                        size / S,
                        core / (C - 1)
                    ])
                else:
                    feats.extend([0.0, 0.0, 0.0])

            feats.append(len(clist) / S)

            # ---- DC blocks ----
            dlist = cache["dc"][ri]

            for j in range(self.j):
                if j < len(dlist):
                    core, start, size = dlist[j]
                    feats.extend([
                        start / S,
                        size / S,
                        core / (C - 1)
                    ])
                else:
                    feats.extend([0.0, 0.0, 0.0])

            feats.append(len(dlist) / S)

            feats.append(
                1.0 if cache["dc_route"][ri]
                is not None else 0.0
            )

            feats.append(
                cache["dc_slots"][ri] / S
            )

           
        # global spectrum condition (from cache)
        feats.append(cache["frag"])
        feats.append(cache["util"])

        src_oh = [0] * n
        dst_oh = [0] * n

        src_oh[
            self.net.nodes.index(request["src"])
        ] = 1

        dst_oh[
            self.net.nodes.index(request["dst"])
        ] = 1

        descriptor = src_oh + dst_oh + [
            request["qc_slots"],
            request["cc_slots"],
            request["demand"]
            / max(self.traffic.capacity_classes)
        ]

        return np.asarray(
            feats + descriptor,
            dtype=np.float32
        )

    # =================================================
    # ACTION MASK (from the same cache)
    # =================================================

    def action_mask(self, request, routes):

        cache = self._ensure_cache(request, routes)

        mask = np.zeros(
            self.action_size,
            dtype=np.float32
        )

        qc_need = request["qc_slots"] + mr.GUARD_BAND
        cc_need = request["cc_slots"] + mr.GUARD_BAND

        adj = self.net.core_adjacency

        for a, (ri, qci, jq, jc, jd) in enumerate(
                self.action_table):

            if ri >= len(routes):
                continue

            if cache["dc_route"].get(ri) is None:
                continue

            qlist = cache["qc"].get((ri, qci), [])
            clist = cache["cc"].get(ri, [])
            dlist = cache["dc"].get(ri, [])

            if jq >= len(qlist):
                continue

            if jc >= len(clist):
                continue

            if jd >= len(dlist):
                continue

            # Intra-request XT check: the QC placement must
            # not forbid the chosen CC block (same route,
            # XT-adjacent cores, overlapping slot ranges).
            # DC runs on a link-disjoint route, so it cannot
            # conflict with QC or CC.
            qcore = self.net.quantum_cores[qci]
            q_start = qlist[jq][1]

            c_core, c_start, _ = clist[jc]

            if adj[qcore][c_core] and \
                    q_start < c_start + cc_need and \
                    c_start < q_start + qc_need:
                continue

            mask[a] = 1.0

        return mask

    # =================================================
    # Blocking diagnosis when mask.sum() == 0, using the
    # same cache. Reasons follow the flowchart labels:
    #   route - no valid link-disjoint route combination
    #   qc / cc / dc - no feasible free block for that
    #                  channel on any candidate route
    #   combination - blocks exist per channel, but no
    #                 jointly feasible (XT-safe) action
    # =================================================

    def mask_block_reason(self, request, routes):

        if not routes:
            return "route"

        cache = self._ensure_cache(request, routes)

        valid = [
            ri for ri in cache["dc_route"]
            if cache["dc_route"][ri] is not None
        ]

        if not valid:
            return "route"

        if not any(
                cache["qc"].get((ri, qci))
                for ri in valid
                for qci in range(
                    len(self.net.quantum_cores))):
            return "qc"

        if not any(
                cache["cc"].get(ri) for ri in valid):
            return "cc"

        if not any(
                cache["dc"].get(ri) for ri in valid):
            return "dc"

        return "combination"

    # =================================================
    # FRAGMENTATION on a route (Section 8 metric)
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
                len(b) for b in blocks
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
    # Flow: Allocate Required Slots inside Selected Block
    # The cached block is re-VERIFIED (not re-generated)
    # before occupying, because the QC placement of the
    # same request updates the XT mask seen by CC / DC.
    # =================================================

    def _occupy_verified(self, link_ids, core,
                         start, required):

        need = required + mr.GUARD_BAND

        occ = mr.feasible_occupancy(
            self.slot_table,
            link_ids,
            core,
            self.net.core_adjacency,
            self.num_training_slots
        )

        if start + need > len(occ):
            return None

        if any(occ[s] for s in
               range(start, start + need)):
            return None

        return mr.occupy(
            self.slot_table,
            link_ids,
            core,
            start,
            required
        )

    # =================================================
    # Flow: Allocate QC + CC + DC (Atomic Allocation)
    # Any failure rolls back everything placed so far.
    # Reward: +1 served, -1 blocked.
    # =================================================

    def step(self, request, routes, action):

        cache = self._ensure_cache(request, routes)

        ri, qci, jq, jc, jd = \
            self.action_table[action]

        info = {
            "served": False,
            "block_reason": None,
           
        }

        if ri >= len(routes):
            info["block_reason"] = "route"
            return -1.0, True, info

        qc_route = routes[ri]

        dc_route = cache["dc_route"].get(ri)

        if dc_route is None:
            info["block_reason"] = "route"
            return -1.0, True, info

        placed = []

        # ---------------- QC ----------------

        qlist = cache["qc"].get((ri, qci), [])

        if jq >= len(qlist):
            info["block_reason"] = "qc"
            return -1.0, True, info

        core, start, _ = qlist[jq]

        idx = self._occupy_verified(
            qc_route, core, start,
            request["qc_slots"]
        )

        if idx is None:
            info["block_reason"] = "qc"
            return -1.0, True, info

        placed.append((qc_route, core, idx))

        # ---------------- CC ----------------

        clist = cache["cc"].get(ri, [])

        if jc >= len(clist):
            self._rollback(placed)
            info["block_reason"] = "cc"
            return -1.0, True, info

        core, start, _ = clist[jc]

        idx = self._occupy_verified(
            qc_route, core, start,
            request["cc_slots"]
        )

        if idx is None:
            self._rollback(placed)
            info["block_reason"] = "cc"
            return -1.0, True, info

        placed.append((qc_route, core, idx))

        # ---------------- DC ----------------

        dlist = cache["dc"].get(ri, [])

        if jd >= len(dlist):
            self._rollback(placed)
            info["block_reason"] = "dc"
            return -1.0, True, info

        core, start, _ = dlist[jd]

        idx = self._occupy_verified(
            dc_route, core, start,
            cache["dc_slots"][ri]
        )

        if idx is None:
            self._rollback(placed)
            info["block_reason"] = "dc"
            return -1.0, True, info

        placed.append((dc_route, core, idx))

        # spectrum changed -> cached block lists are stale
        self._cache = None

        self.active[request["id"]] = {
            "release_time":
                request["arrival"]
                + request["holding"],
            "placements": placed
        }

        info["served"] = True
        info["block_reason"] = "success"
        

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
    # Flow: release expired requests so freed slots
    # merge back into new free blocks.
    # =================================================

    def release_expired(self, current_time):

        released = False

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

                released = True

        if released:
            # spectrum changed -> cached block lists are stale
            self._cache = None

    # =================================================

    def reset(self):

        mr.reset_slot_table(self.slot_table)

        self.active.clear()

        self._cache = None

        self._reset_iter()

        