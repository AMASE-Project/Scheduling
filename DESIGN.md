# AMASE-P Scheduling — Design Document

Technical design of the AMASE-P observing scheduler: architecture,
time model, constraints, target model, visibility precomputation, the
nightly MILP, the multi-night campaign engine, and the output layer.
For the user-facing manual see `README.md`; implementation modules live
in `src/amase_scheduling/`.

---

## 1. Architecture overview

Three command-line tools form a file-driven pipeline. Each tool does one
job; CSV and `.npz` files are the only interfaces between them.

```
amase-precompute  →  vis_cache.npz            (visibility, parallel)
amase-schedule    →  schedule.csv             (blocks)
                     schedule_targets.csv     (per-target progress)
                     schedule_nights.csv      (nightly index)
amase-plot        →  figures from those CSVs + the original target list
```

### 1.1 Module map

| Module | Responsibility |
|--------|----------------|
| `observatory.py` | Nanshan site parameters; `night_window()` (sunset → next sunrise) |
| `constraints.py` | Constraint constants (30° / −12° / 0.25 / 30°) and checkers; Moon illumination |
| `visibility.py` | 5-min slot grid; vectorized per-slot visibility; valid-start sliding window; transit-quality factor; `dark_window()` |
| `target.py` | `Target` dataclass; strict CSV loading and validation; sexagesimal normalization; `invert_priorities()` |
| `milp.py` | `build_milp` (x/y/c variables, five constraint families, three-term objective); `solve_milp` (CBC invocation and decoding) |
| `cache.py` | `VisibilityCache`: per-night `NightVisibility` storage; parallel `build`; `.npz` persistence; target-list fingerprint validation |
| `scheduler.py` | **Unified engine** `Scheduler.schedule` (single night = degenerate campaign) plus the `Schedule / NightPlan / ScheduledBlock / TargetProgress` dataclasses; capacity warning |
| `weather.py` | `WeatherModel`: Bernoulli per-night clear probability with seeded RNG; `clear_prob = 1` means always clear |
| `output.py` | Unified console report `format_report`; CSV trio writers; `load_schedule_csvs` (rebuilds a `Schedule` from the CSV trio) |
| `plotting.py` | All figures: two-panel night figure, two-panel campaign figure, single-target track diagnostic; batch night-figure export |
| `cli.py` | `amase-schedule` entry point |
| `cli_precompute.py` | `amase-precompute` entry point |
| `cli_plot.py` | `amase-plot` entry point (`campaign` / `night` / `nights` / `track`) |

### 1.2 Design principles

- **Optimality first**: MILP (PuLP + CBC) guarantees the optimal plan
  under the stated constraints.
- **One engine**: a single night is not a special case — it is a
  1-night campaign with `clear_prob = 1.0`. All code paths are shared.
- **Astronomy before optimization**: every astronomy-dependent
  constraint is evaluated in the visibility precomputation; the MILP
  sees only a Boolean matrix.
- **File-based decoupling**: scheduling knows nothing about plotting;
  the CSV trio is a complete, documented interchange format, so figures
  can be restyled without re-running the optimizer.

---

## 2. Observatory and time model

| Parameter | Value |
|-----------|-------|
| Site | Nanshan Observatory, Xinjiang |
| Longitude | 87.1750°E |
| Latitude | 43.4720°N |
| Elevation | 2080 m |
| Timezone | UTC internally; UTC+8 shown in human-readable outputs |

### 2.1 Night window

For each date, the scheduling window runs from **sunset to the next
sunrise** (`observatory.night_window`, astroplan root-finding). Two
deliberate choices:

1. The window is sunset→sunrise (Sun < 0°), *not* the −12° twilight
   bounds; the Sun < −12° condition is applied **per slot** in the
   visibility matrix, so the window is a superset of usable time and
   twilight slots are filtered out automatically.
2. Sunset is found first, then the following sunrise, guaranteeing
   `night_end > night_start` (otherwise winter dates could yield a
   negative night length).

### 2.2 Slots

The window is discretized into uniform **slots of δ = 5 minutes**
(`visibility.slot_times`). Slot `k` covers `[t_k, t_k + δ)`. A summer
night has T ≈ 105 slots, a winter night T ≈ 180. Visibility is sampled
at slot start times.

### 2.3 Blocks and overhead

An observation of target `i` is an indivisible **block**:

```
block_sec(i) = exp_time_i × n_dither_i        (dithers are back-to-back)
B_i = max(1, ceil(block_sec(i) / δ))          (block length in slots)
O   = 2                                        (10 min overhead)
```

After every block the telescope needs O = 2 slots (10 min) of overhead
(slew + settle) during which no other block may start. A block starting
at slot `k` therefore occupies `[k, k + B_i + O)`, of which `B_i` slots
are productive. The trailing overhead slots need not be visible — the
telescope may point anywhere while slewing.

---

## 3. Observing constraints

| Constraint | Rule |
|------------|------|
| Altitude | target alt > 30° at slot time |
| Sun | Sun alt < −12° (nautical darkness) at slot time |
| Moon | passes if **any** of: illumination < 0.25; Moon below horizon; Moon–target separation > 30° |
| Overhead | 10 min after every block (see §2.3) |

The Moon rule is a three-way disjunction — in particular, a bright Moon
below the horizon never vetoes a target.

---

## 4. Target model and input validation

### 4.1 Target fields

| Field | Meaning |
|-------|---------|
| `name` | Target name |
| `ra`, `dec` | Sky position (decimal degrees or sexagesimal) |
| `priority` | Static priority weight; larger wins. Rank-style catalogs (1 = highest) are converted with `invert_priorities()` / `--invert-priority` (w = p_max + p_min − p) |
| `exp_time` | Single exposure time (s) |
| `n_dither` | Dither count per visit (back-to-back, indivisible) |
| `n_exposure` | **Season-total** visit demand; carried across nights, decremented per scheduled block; targets at zero graduate |
| `group` | Grouping label for plot coloring/statistics; defaults to `Untitle` |

Derived quantities: `block_duration_sec = exp_time × n_dither`;
`total_time_sec = block × n_exposure`.

### 4.2 Input validation (`target.py`)

CSV only (8 columns, case-insensitive header). Row-numbered errors are
raised for:

- any missing column or empty cell except `group` (missing/empty group
  → `Untitle`);
- `ra`/`dec` not in one of the two accepted forms — decimal degrees, or
  sexagesimal (`12h18m57.5s`, `12:18:57.5`, `+47d18m14s`); fractional
  hours/minutes (`05h40.9m0.0s`) are normalized by carrying into the
  next lower unit before parsing;
- `exp_time` outside `(0, 3600] s`;
- `n_dither` not in `{1, 3, 9, 27}`;
- `n_exposure < 1`.

`priority` is unconstrained (any float).

---

## 5. Visibility precomputation (`visibility.py`)

All astronomical constraints are evaluated before the MILP, so the
optimizer never sees an astronomy-dependent constraint.

### 5.1 Per-slot visibility

`compute_visibility` evaluates, for every target `i` and slot `k`:

```
visible[i][k] = ( alt_i(k) > 30° )
              ∧ ( sun_alt(k) < −12° )
              ∧ ( illum(k) < 0.25
                ∨ moon_alt(k) < 0°
                ∨ moon_sep_i(k) > 30° )
```

Sun and Moon positions are computed once per night as vectorized time
arrays; Moon illumination uses the elongation approximation
`(1 − cos ψ)/2`, likewise vectorized. Per-target work is then a single
coordinate transform plus a separation.

### 5.2 Valid start slots

A block starting at slot `k` needs its `B_i` **observing** slots all
visible (the trailing overhead slots may fall anywhere):

```
valid_start[i][k] = (k + B_i ≤ T) ∧ all( visible[i][k : k+B_i] )
```

computed by sliding a window of length `B_i` over each visibility row.
Entries with `valid_start = False` never create MILP variables, which
compresses the variable space substantially. A target with no valid
start at all is **unschedulable** tonight (transit inside twilight, or
longest visible stretch shorter than one block) and is excluded from
the model before building.

### 5.3 Transit-quality factor

For the time-of-night preference, each valid start slot carries

```
q(i, k) = sin( alt_i at block midpoint of start k ) / sin( alt_max_i tonight )   ∈ (0, 1]
```

- the **block midpoint** altitude represents the whole block better
  than the start time;
- normalization is **per target** (an airmass ratio): `q = 1` at the
  target's own best moment tonight, so low-declination targets are not
  systematically discriminated against;
- `q` is an objective weight, not a feasibility criterion. The altitude
  matrix computed for `visible` makes `q` essentially free, and it is
  stored in the cache alongside `valid_start`.

---

## 6. The nightly MILP (`milp.py`)

### 6.1 Notation

| Symbol | Meaning |
|--------|---------|
| `N` | number of schedulable targets (unschedulable ones removed) |
| `T` | number of slots tonight |
| `K_i` | visits still required for target `i` (campaign: remaining; single night: `n_exposure_i`) |
| `B_i` | block length in slots; `O = 2` overhead slots |
| `w_i` | priority weight |
| `ε`, `γ` | small bonus weights (defaults 1e-3, 1e-2) |
| `α` | transit-preference strength ∈ [0, 1] (default 0.5) |

### 6.2 Decision variables

| Variable | Meaning | Created when |
|----------|---------|--------------|
| `x[i][k] ∈ {0,1}` | one block of target `i` starts at slot `k` | only if `valid_start[i][k]` |
| `y[i] ∈ {0,1}` | target `i` gets ≥ 1 block tonight | always |
| `c[i] ∈ {0,1}` | all `K_i` remaining blocks of target `i` fit tonight | if `γ > 0` and the target has variables |

**Merging over the exposure index.** Blocks of one target are
physically identical (same duration, same reward), so the model does
not distinguish "the j-th visit". Variables `x[i][j][k]` would create
`K_i!` symmetric copies of every schedule, sending branch-and-bound
wandering through equivalent permutations. Instead `x[i][k]` records
*whether a block starts at slot k*, with the visit count enforced by
the capacity constraint. This cuts the variable count ≈ 3× and speeds
up solving ≈ 8× (measured: 300 targets, winter night, 9.2 s → 1.1 s).
The `visit` numbers in the output are assigned chronologically during
decoding — display labels only.

### 6.3 Constraints

1. **Capacity** (`cap_i`): at most `K_i` blocks per target
   `∀i: Σ_k x[i][k] ≤ K_i`
2. **Slot exclusivity** (`slot_t`) — the core constraint; at any time
   at most one block occupies the telescope, including its overhead
   tail:
   `∀t: Σ_{i,k : k ≤ t < k + B_i + O} x[i][k] ≤ 1`
   Built with an inverted index: each variable registers itself only
   into the `B_i + O` slots it covers (clipped at `T` near dawn),
   instead of scanning all variable–slot pairs (≈ 3× faster model
   build).
3. **Linking** (`link_i`): `Σ_k x[i][k] ≥ y[i]` — `y` may be 1 only if
   a block exists. (The reverse direction is unnecessary: the `+ε·y[i]`
   term raises `y[i]` freely once a block exists.)
4. **No viable start** (`nosched_i`): a target with no valid start slot
   has `y[i] = 0` fixed and no `x` variables at all.
5. **Completion** (`complete_i`, only if `γ > 0`):
   `Σ_k x[i][k] ≥ K_i · c[i]`
   Together with capacity this clamps `Σ_k x[i][k] = K_i` whenever
   `c[i] = 1`; the `+γ·c[i]` reward makes the solver set `c[i] = 1`
   exactly when all remaining blocks fit.

### 6.4 Objective function

```
max   Σ_{i,k}  w_i · B_i · [(1 − α) + α · q(i,k)] · x[i][k]     (main term)
    + ε · Σ_i y[i]                                              (diversity bonus)
    + γ · Σ_i c[i]                                              (completion bonus)
```

**Main term.** Reward is proportional to block duration, so the
objective is effectively "priority-weighted allocated telescope time".
Long blocks are naturally efficient: a block occupies `B_i + O` slots
to deliver `B_i` productive ones, a fraction `B_i/(B_i + O)` (92% for
a 110-min block, 33% for a 5-min one). Overhead amortization is thus
encoded economically rather than as extra constraints — the model
prefers few long slews, matching real observing practice.

**Transit weighting α.** `q(i,k)` scales each block's reward between
`(1−α)·w_i·B_i` (away from transit) and `w_i·B_i` (at transit). This is
a *soft* preference: if the transit window is taken, a target can still
be scheduled off-transit at slightly reduced reward — graceful
degradation instead of hard time windows. Measured effect (α = 0.6):
mean offset from transit 143 → 104 min at unchanged total allocated
time; as a side benefit, the quality gradient breaks ties and CBC
converges *faster*.

**ε and γ** are orders of magnitude below any main-term reward, so they
only break ties — towards covering more distinct targets (ε) and
towards finishing a target's remaining visits in one night (γ).
Neither can overturn a clearly better main-term solution. (Verified
safe for weights w ∈ {1, 2, 3} with the default ε, γ; it is the `w·B`
scale, not the bonuses, that drives long-block dominance.)

### 6.5 Worked example

T = 10 slots, O = 2, ε = 0.001, γ = 0.01. Three targets:

| Target | w | B | K | visible slots |
|--------|---|---|---|---------------|
| A      | 5 | 2 | 1 | 0–3 |
| B      | 8 | 1 | 2 | 2–5 |
| C      | 3 | 2 | 1 | 5–9 |

Valid starts: A ∈ {0,1}, B ∈ {2,3,4,5}, C ∈ {5,6,7}.

**Optimal solution**: A@0 (occupies [0,4)), B@4 (occupies [4,7)),
C@7 (occupies [7,11); the trailing overhead beyond slot 10 is clipped
harmlessly).

```
main term:   5·2 + 8·1 + 3·2 = 24
diversity:   ε·3             = 0.003
completion:  γ·(c_A + c_C)   = 0.02     (B got only 1 of 2 blocks → c_B = 0)
total: 24.023
```

The runner-up (A@1, B@5) scores 18 + 2ε + γ = 18.012 — strictly worse.

Decoded observing sequence (slot 0 = 18:00 UTC):

```
UTC start  UTC end   Duration  Target
18:00      18:10     10 min    A     (telescope busy until 18:20 incl. overhead)
18:20      18:25      5 min    B
18:35      18:45     10 min    C
```

---

## 7. Solving and decoding

- PuLP builds the model; **CBC** (branch-and-cut) solves it
  (`time_limit` default 60 s per night).
- The solver status is `Optimal` (proven) or `Feasible` (time limit
  hit; best incumbent returned — all constraints still satisfied).
  Anything else yields an empty plan (the all-zero solution is always
  feasible, so true infeasibility cannot occur).
- Decoding: each `x[i][k] = 1` becomes a block `[t_k, t_k + B_i·δ)`;
  altitude, azimuth and Moon separation at the block midpoint are
  evaluated per block for the report; blocks are emitted in
  chronological order.

Typical size (300 targets, winter night): ≈ 13k binary variables,
≈ 1k constraints, solved to proven optimality in ≈ 1 s.

---

## 8. Multi-night campaign engine (`scheduler.py`)

### 8.1 State semantics

`n_exposure` is the **season-total demand**. A `remaining[i]` counter
carries across nights, decremented once per scheduled block; targets
hitting zero graduate and no longer enter subsequent nights' models.
Visits of one target carry no minimum-gap constraint; priorities are
static.

### 8.2 Weather

Each night is independently clear with probability `clear_prob`
(Bernoulli, seeded RNG → fully reproducible for fixed `--seed`). A
cloudy night contributes an empty plan; equivalently, its visibility
matrix would be all zeros. `clear_prob = 1` disables weather entirely
(the single-night default).

### 8.3 Unified engine loop

```python
def schedule(targets, start, end=None, clear_prob=1.0, seed=None, ...):
    remaining = [t.n_exposure for t in targets]
    weather   = WeatherModel(clear_prob, seed)
    if n_nights > 1: capacity_check(...)
    for date in nights(start, end or start):     # serial: night N+1 depends on N
        if not weather.is_clear():
            record cloudy night; continue
        active = [i for i in targets if remaining[i] > 0]
        night  = solve_one_night(active, date, k_remaining=remaining[active])
        for block in night.blocks:
            remaining[block.target] -= 1         # state update
            accumulate progress
    return Schedule(nights, progress)
```

- **Single-night degeneracy**: one date, `clear_prob = 1.0`,
  `remaining = n_exposure` — the same code path, no special branches.
- **Capacity warning** (multi-night only): total demand
  `Σ exp_time·n_dither·n_exposure` vs expected available time
  `n_nights × mean night length × clear_prob`, reported up front when
  the campaign is oversubscribed.

### 8.4 Why per-night decomposition is sound

A single season-long MILP would be enormous and is unnecessary:

- **Blocks never span nights** — daylight separates nights absolutely.
- **The objective is night-separable** given `remaining` — tonight's
  reward does not depend on when other nights' blocks fall.
- **No inter-night constraints** exist in the problem specification
  (no minimum gap between visits of one target, no deadlines).

Hence a sequence of per-night MILPs is equivalent to one global MILP
under deterministic weather. The only suboptimality comes from the
greedy allocation of `remaining` across nights, which is negligible
under static priorities without due dates.

### 8.5 Visibility cache and parallelism

Visibility is independent of weather and state, so it can be
**precomputed in parallel across nights** (`amase-precompute`,
multiprocessing) and persisted to disk (`.npz`: `valid_start`,
`quality`, slot times, window bounds). `amase-schedule --cache` reuses
it across any number of simulations with different seeds or tuning
parameters; a fingerprint of the target list (names + order) plus the
date range guards against stale caches, with a clear error on mismatch.
After caching, the nightly MILP is the dominant cost.

---

## 9. Output layer

### 9.1 Unified console report (`output.format_report`)

One report structure for any campaign length:

1. header (period, weather model, capacity warning);
2. night section — single night expands into the full block table
   (per-block start/end UTC, duration, altitude, azimuth, Moon
   separation, plus window/efficiency/unscheduled lists); multi-night
   uses a compact per-night summary table;
3. target completion table (sorted by fraction; truncated to the 20
   most complete with an omission note, full table in the CSV);
4. totals (clear nights, completed/partial/untouched targets, visits,
   total observing time).

### 9.2 The CSV trio (interchange format)

Written by `amase-schedule -o FILE`:

| File | Columns | Content |
|------|---------|---------|
| `FILE` (blocks) | `date, target, visit, obs_start_utc, obs_end_utc, obs_start_local, obs_end_local, lst_start, lst_end, duration_min, altitude_deg, azimuth_deg, moon_sep_deg` | one row per scheduled block; LST = apparent sidereal time, HH:MM mod 24 |
| `<stem>_targets.csv` | `target, required, done, fraction, nights_observed, obs_hours` | per-target campaign progress |
| `<stem>_nights.csv` | `date, clear, night_start_utc, night_end_utc, dark_start_utc, dark_end_utc, n_blocks, n_targets, obs_hours, n_unschedulable` | nightly index; window columns are physical and filled for cloudy nights too, distinguishing weather loss from clear-but-empty nights |

`output.load_schedule_csvs` rebuilds a `Schedule` from the trio (+ the
original target list for coordinates/groups); this is the sole input to
`amase-plot`, so plotting never re-runs scheduling.

### 9.3 Figures (`plotting.py`)

- **Night figure** (two panels): altitude tracks with scheduled blocks
  bolded, 30° limit, twilight shading; Gantt timeline with overhead
  tails. Gray context tracks for unscheduled targets are drawn only for
  lists of ≤ 100 targets.
- **Campaign figure** (two panels): night-window utilization (x = date,
  y = local UTC+8; gray sunset→sunrise bands for clear nights, dashed
  −12° bounds for all nights, blocks colored by group, cloudy nights
  blank) and completion bars — per-target clustered by group for ≤ 40
  targets, one aggregated bar per group above. Group colors use a
  fixed tab20-derived palette.
- **Track figure** (two panels): single-target altitude and
  Moon-separation diagnostics with the Moon-constraint-passing
  intervals shaded green; title shows the night's Moon illumination.
  The site is a parameter (default Nanshan).

---

## 10. Edge cases

| Situation | Handling |
|-----------|----------|
| No target schedulable tonight | empty plan; all names reported as unscheduled |
| Target's `B_i` exceeds its longest visible stretch | `valid_start` all False → excluded and reported |
| Night too short for all demand | main term automatically prefers high `w·B` combinations; low-value targets drop out |
| All priorities equal | long blocks win on efficiency; ε favors covering more targets |
| Target completed mid-campaign | `remaining = 0` → excluded from later nights |
| Clear night, everything done | recorded as an empty plan |
| Cloudy night | empty plan with `clear = False`; physical windows still written to the nights CSV |

---

*User manual: `README.md`. API walkthrough: `example/demo.py`.*
