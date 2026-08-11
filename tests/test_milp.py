import numpy as np

from amase_scheduling.milp import OVERHEAD_SLOTS, build_milp, solve_milp


def _solve(valid, B, K, w):
    prob, var_map, solver = build_milp(
        valid, B, K, w, gamma=0.0, time_limit=30
    )
    return solve_milp(prob, var_map, solver)


def test_chain_packs_same_target_back_to_back():
    # T=6, one target, B=1, K=3. With chaining, 3 exposures fit (two of
    # them must be adjacent, waiving the overhead between them).
    # Without chaining (pitch B+O=3) at most 2 would fit.
    valid = np.ones((1, 6), dtype=bool)
    sched = _solve(valid, np.array([1]), np.array([3]), np.array([1.0]))
    assert len(sched) == 3
    starts = sorted(k for _, k in sched)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert min(gaps) == 1


def test_target_switch_still_charges_overhead():
    # T=4, two targets, B=1, K=1 each. Different targets cannot chain,
    # so the only feasible packing is starts {0, 3}.
    valid = np.ones((2, 4), dtype=bool)
    sched = _solve(valid, np.array([1, 1]), np.array([1, 1]), np.array([1.0, 1.0]))
    assert len(sched) == 2
    starts = sorted(k for _, k in sched)
    assert starts[1] - starts[0] >= 1 + OVERHEAD_SLOTS


def test_mixed_run_beats_interleaving():
    # T=6, targets A,B with B=1, K=2 each. Chained runs AA+BB fit with
    # room to spare (AA oh, BB): the optimum schedules all 4 exposures.
    valid = np.ones((2, 6), dtype=bool)
    sched = _solve(
        valid, np.array([1, 1]), np.array([2, 2]), np.array([1.0, 1.0])
    )
    assert len(sched) == 4
