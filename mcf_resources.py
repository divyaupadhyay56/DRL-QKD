import random

GUARD_BAND = 1


# ==========================================================
# SLOT TABLE
# ==========================================================
def build_slot_table(num_links, cores_per_link, slots_per_core):
    return [[[0] * slots_per_core
             for _ in range(cores_per_link)]
             for _ in range(num_links)]


def reset_slot_table(slot_table):
    for link in slot_table:
        for core in link:
            for s in range(len(core)):
                core[s] = 0


# ==========================================================
# FREE BLOCKS
# ==========================================================
def free_blocks(occupancy):
    blocks = []
    current = []

    for s, busy in enumerate(occupancy):

        if busy == 0:
            current.append(s)

        elif current:
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)

    return blocks


# ==========================================================
# ROUTE OCCUPANCY
# ==========================================================
def route_core_occupancy(slot_table, link_ids, core):

    width = len(slot_table[link_ids[0]][core])

    merged = [0] * width

    for lid in link_ids:

        vec = slot_table[lid][core]

        for s in range(width):
            merged[s] = merged[s] or vec[s]

    return merged


# ==========================================================
# XT AVOIDANCE
# ==========================================================
def xt_forbidden(slot_table,
                 link_ids,
                 core,
                 core_adjacency):

    width = len(slot_table[link_ids[0]][core])

    forbidden = [0] * width

    neighbours = [
        c
        for c in range(len(core_adjacency))
        if core_adjacency[core][c]
    ]

    for lid in link_ids:

        for n in neighbours:

            nvec = slot_table[lid][n]

            for s in range(width):

                if nvec[s]:
                    forbidden[s] = 1

    return forbidden


# ==========================================================
# TRAINING SLOT LIMIT
# ==========================================================
def feasible_occupancy(slot_table,
                       link_ids,
                       core,
                       core_adjacency,
                       max_slots=None):

    occ = route_core_occupancy(
        slot_table,
        link_ids,
        core
    )

    forb = xt_forbidden(
        slot_table,
        link_ids,
        core,
        core_adjacency
    )

    result = []

    for s in range(len(occ)):

        if max_slots is not None:

            if s >= max_slots:
                result.append(1)
                continue

        result.append(
            occ[s] or forb[s]
        )

    return result


# ==========================================================
# CANDIDATE BLOCKS
# ==========================================================
def _candidate_blocks(occupancy, required):

    need = required + GUARD_BAND

    return [
        b
        for b in free_blocks(occupancy)
        if len(b) >= need
    ]


# ==========================================================
# FIT STRATEGIES
# ==========================================================
def select_start(occupancy,
                 required,
                 strategy):

    blocks = _candidate_blocks(
        occupancy,
        required
    )

    if not blocks:
        return None

    if strategy == "first":
        return blocks[0][0]

    if strategy == "best":

        best = min(
            blocks,
            key=lambda b: (len(b), b[0])
        )

        return best[0]

    if strategy == "random":

        chosen = random.choice(blocks)

        max_offset = (
            len(chosen)
            - (required + GUARD_BAND)
        )

        return chosen[
            random.randint(0, max_offset)
        ]

    raise ValueError(
        "Unknown strategy"
    )


# ==========================================================
# OCCUPY
# ==========================================================
def occupy(slot_table,
           link_ids,
           core,
           start,
           required):

    indexes = list(
        range(
            start,
            start + required + GUARD_BAND
        )
    )

    for lid in link_ids:

        for s in indexes:

            slot_table[lid][core][s] = 1

    return set(indexes)


# ==========================================================
# RELEASE
# ==========================================================
def release(slot_table,
            link_ids,
            core,
            indexes):

    for lid in link_ids:

        for s in indexes:

            slot_table[lid][core][s] = 0


# ==========================================================
# EXTERNAL FRAGMENTATION
# ==========================================================
def external_fragmentation(slot_table):

    ratios = []

    for link in slot_table:

        for core in link:

            blocks = free_blocks(core)

            total_free = sum(
                len(b)
                for b in blocks
            )

            if total_free == 0:

                ratios.append(0.0)

            else:

                largest = max(
                    len(b)
                    for b in blocks
                )

                ratios.append(
                    1.0 - largest / total_free
                )

    if not ratios:
        return 0.0

    return sum(ratios) / len(ratios)


# ==========================================================
# UTILISATION
# ==========================================================
def spectrum_utilisation(slot_table):

    used = 0
    total = 0

    for link in slot_table:

        for core in link:

            used += sum(core)

            total += len(core)

    if total == 0:
        return 0.0

    return used / total