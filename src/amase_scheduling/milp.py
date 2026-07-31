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
) -> tuple[pulp.LpProblem, dict[tuple[int, int], pulp.LpVariable], pulp.PULP_CBC_CMD]:
    """Build the scheduling MILP.

    Variables are merged over the exposure index j: x[i,k]=1 means one block
    of target i starts at slot k. Since a slot admits at most one block, the
    per-j binaries would only add K_i! symmetric copies of every solution;
    the cap sum_k x[i,k] <= K_i enforces the visit count instead.

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

        y_i = pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary)
        y_vars.append(y_i)

        x_list = []
        for k in valid_k:
            var = pulp.LpVariable(f"x_{i}_{k}", cat=pulp.LpBinary)
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
            c_i = pulp.LpVariable(f"c_{i}", cat=pulp.LpBinary)
            c_vars.append(c_i)
            prob += pulp.lpSum(x_list) >= K * c_i, f"complete_{i}"

    prob += (
        pulp.lpSum(block_terms)
        + eps * pulp.lpSum(y_vars)
        + gamma * pulp.lpSum(c_vars)
    ), "Objective"

    slot_vars: list[list[pulp.LpVariable]] = [[] for _ in range(T)]
    B_list = [int(b) for b in block_slots]
    for (i, k), var in var_map.items():
        t_end = k + B_list[i] + OVERHEAD_SLOTS
        if t_end > T:
            t_end = T
        for t in range(k, t_end):
            slot_vars[t].append(var)
    for t in range(T):
        if slot_vars[t]:
            prob += pulp.lpSum(slot_vars[t]) <= 1, f"slot_{t}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    return prob, var_map, solver


def solve_milp(
    prob: pulp.LpProblem,
    var_map: dict[tuple[int, int], pulp.LpVariable],
    solver: pulp.PULP_CBC_CMD,
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
