import numpy as np
import pulp

OVERHEAD_SLOTS = 2


def build_milp(
    valid_start: np.ndarray,
    block_slots: np.ndarray,
    n_exposure: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-3,
    gamma: float = 0.0,
    time_limit: int = 60,
    quality: np.ndarray | None = None,
    alpha: float = 0.0,
) -> tuple[pulp.LpProblem, dict[tuple[int, int], pulp.LpVariable], pulp.COIN_CMD]:
    """Build the scheduling MILP.

    Variables are merged over the exposure index j: x[i,k]=1 means one block
    of target i starts at slot k. Since a slot admits at most one block, the
    per-j binaries would only add K_i! symmetric copies of every solution;
    the cap sum_k x[i,k] <= K_i enforces the exposure count instead.

    Slew overhead (OVERHEAD_SLOTS) is charged only on a target switch:
    chain[i,k]=1 iff blocks of target i start at both k and k+B_i, in which
    case the post-block overhead is waived. The slot constraint counts
    (x[i,k] - chain[i,k]) over the overhead window [k+B_i, k+B_i+2).
    chain needs only upper bounds chain <= x[i,k], chain <= x[i,k+B_i]:
    it carries no objective weight and only relaxes <= constraints, so the
    solver pushes it to min(x[i,k], x[i,k+B_i]) automatically.

    If `quality` (N×T, same row order as valid_start) and alpha > 0 are
    given, each block's objective coefficient is time-weighted toward the
    target's transit: coeff = w*B*[(1-alpha) + alpha*q(i,k)].
    """
    N, T = valid_start.shape
    prob = pulp.LpProblem("AMASE_Scheduling", pulp.LpMaximize)

    var_map: dict[tuple[int, int], pulp.LpVariable] = {}
    y_vars: list[pulp.LpVariable] = []
    c_vars: list[pulp.LpVariable] = []
    block_terms = []

    for i in range(N):
        B = int(block_slots[i])
        K = int(n_exposure[i])
        w = float(weights[i])
        valid_k = np.where(valid_start[i])[0]

        y_i = prob.add_variable(f"y_{i}", cat=pulp.LpBinary)
        y_vars.append(y_i)

        x_list = []
        for k in valid_k:
            var = prob.add_variable(f"x_{i}_{k}", cat=pulp.LpBinary)
            var_map[(i, k)] = var
            x_list.append(var)
            if quality is not None and alpha > 0:
                coeff = w * B * ((1.0 - alpha) + alpha * float(quality[i, k]))
            else:
                coeff = w * B
            block_terms.append(coeff * var)

        if x_list:
            prob += pulp.lpSum(x_list) >= y_i, f"link_{i}"
            prob += pulp.lpSum(x_list) <= K, f"cap_{i}"
        else:
            prob += y_i == 0, f"nosched_{i}"

        if gamma > 0 and K > 0 and x_list:
            c_i = prob.add_variable(f"c_{i}", cat=pulp.LpBinary)
            c_vars.append(c_i)
            prob += pulp.lpSum(x_list) >= K * c_i, f"complete_{i}"

    prob += (
        pulp.lpSum(block_terms)
        + eps * pulp.lpSum(y_vars)
        + gamma * pulp.lpSum(c_vars)
    ), "Objective"

    B_list = [int(b) for b in block_slots]

    chain_map: dict[tuple[int, int], pulp.LpVariable] = {}
    for i in range(N):
        B = B_list[i]
        valid_set = set(np.where(valid_start[i])[0].tolist())
        for k in valid_set:
            if (k + B) in valid_set:
                ch = prob.add_variable(f"chain_{i}_{k}", cat=pulp.LpBinary)
                chain_map[(i, k)] = ch
                prob += ch <= var_map[(i, k)], f"chain_a_{i}_{k}"
                prob += ch <= var_map[(i, k + B)], f"chain_b_{i}_{k}"

    slot_terms: list[list] = [[] for _ in range(T)]
    for (i, k), var in var_map.items():
        B = B_list[i]
        for t in range(k, min(k + B, T)):
            slot_terms[t].append(var)
        oh_end = min(k + B + OVERHEAD_SLOTS, T)
        for t in range(k + B, oh_end):
            slot_terms[t].append(var)
            ch = chain_map.get((i, k))
            if ch is not None:
                slot_terms[t].append(-ch)
    for t in range(T):
        if slot_terms[t]:
            prob += pulp.lpSum(slot_terms[t]) <= 1, f"slot_{t}"

    solver = pulp.COIN_CMD(msg=False, timeLimit=time_limit)
    return prob, var_map, solver


def solve_milp(
    prob: pulp.LpProblem,
    var_map: dict[tuple[int, int], pulp.LpVariable],
    solver: pulp.COIN_CMD,
) -> list[tuple[int, int]]:
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Feasible"):
        return []

    schedule = []
    for (i, k), var in var_map.items():
        if pulp.value(var) is not None and pulp.value(var) > 0.5:
            schedule.append((i, k))

    schedule.sort(key=lambda x: x[1])
    return schedule
