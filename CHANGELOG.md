# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`amase-web` can load a precomputed visibility cache**: pick or
  drag-and-drop a `.npz` file in the browser (`POST /api/cache/upload`,
  kept in memory under a `cache_id`), or enter a server-side path
  (`POST /api/cache/load`); `POST /api/schedule` accepts either
  `cache_id` or `cache_path`. The date range is validated against the
  cache coverage up front (400 with a clear message instead of a
  mid-run failure). The frontend *Visibility cache* card shows load
  status, warns on target-list mismatch, and limits the date pickers
  to the cache coverage. `VisibilityCache.load_bytes()` supports the
  upload path.
- **`VisibilityCache` exposure-time fingerprint**: caches now store the
  per-target block-slot counts and `validate()` compares them, so
  editing `exp_time` after building a cache is rejected instead of
  silently using stale visibility. Old-format caches without the
  fingerprint still work but emit a `UserWarning` (names-only check).
  The shipped `example/vis_cache.npz` was regenerated with the
  fingerprint.
- Tests: `tests/test_cache.py` (fingerprint save/load/validate,
  old-format warning) and `tests/test_web_cache.py` (load endpoint,
  coverage and mismatch rejection).
- **`amase-web`** — local web UI (FastAPI backend + static frontend):
  submit campaigns from the browser, watch live progress, browse night
  figures, and stop the server from the page. Runs at
  http://127.0.0.1:8765 by default (`--host` / `--port` / `--no-browser`).
  Scheduler gains a `progress_callback` hook (c5dec92).
- Shipped **`example/vis_cache.npz`**: 2-year visibility cache
  (2027-04-01 .. 2029-04-01) for the 42-target example, so demos and the
  web UI schedule instantly without precompute (d39797a).
- Pipeline overview figure embedded in the README (080568a).
- FAQ entry: far-southern targets are schedulable at exposure
  granularity (1347f88).

### Changed

- New core dependency `python-multipart` (required by the cache upload
  endpoint).
- **Migrated to the PuLP 4.0 API**: `pulp.COIN_CMD` solver and
  `prob.add_variable(...)`; dependency floor raised to `pulp[cbc]>=3.1`
  (d354f76).
- **Web dependencies are now mandatory** (`fastapi`, `uvicorn[standard]`
  in core dependencies); the `.[web]` extra is removed (00e4e8e).
- Night figures show **Nanshan local time (UTC+8)** on the primary
  axis ("Local time (UTC+8)"); the LST secondary axis is unchanged
  (90278a3).
- CLI entry points suppress benign far-future date warnings (astropy
  polar-motion and ERFA "dubious year"); library use is unaffected
  (d461d6c).
- `example/demo.py` loads the shipped cache (with build fallback);
  `demo.ipynb` updated to v0.2.0 terminology and executed clean
  (d39797a).

### Fixed

- Campaign figure date labels no longer overlap (AutoDateLocator with
  30° rotation) (90278a3).
- `demo.ipynb` used the stale attribute `t.n_exposure`; now
  `t.n_exposures` (d39797a).

## [0.2.0] - 2026-08-11

### Changed

- **Schedulable unit is now a single exposure** (`block = exp_time`);
  dither sets are no longer kept contiguous. The season demand is
  `n_dither × n_set` independent exposures, schedulable at any time on
  any night. Finer granularity packs narrow visibility windows that a
  whole dither sequence could not fit.
- **Overhead is charged only on target switches**: same-target adjacent
  exposures chain with zero gap. New `chain[i,k]` binaries waive the
  2-slot slew tail; the slot-exclusivity constraint counts
  `x[i,k] − chain[i,k]` over the overhead window (chain needs only
  upper bounds — the solver maximizes it for free).
- **Input column renamed `n_exposure` → `n_set`** (number of dither
  sets). New derived quantity `Target.n_exposures = n_dither × n_set`;
  `block_duration_sec = exp_time`.
- Output wording visits → exposures (blocks CSV column `exposure`,
  report summary, campaign figure axis); capacity warning adds a
  per-set slew estimate.
- Campaign results are **not comparable with 0.1.0** and visibility
  caches must be rebuilt. On the 42-target example (clear 0.5,
  seed 42): half-year 30 → 35 targets completed, full year 37 → 41
  (only CentaurusA remains unschedulable), delivered time 403 → 494 h.

### Added

- LST start/end columns in the blocks CSV and LST axes on figures
  (0dca49d).
- `tests/test_milp.py`: chain-semantics tests (same-target packing,
  switch overhead, mixed runs).

## [0.1.0] - 2026-07-30

Initial release.

### Added

- **`amase-schedule`** — MILP-based (PuLP + CBC) priority-weighted optimal
  scheduling for the AMASE-P telescope at Nanshan Observatory. Unified
  engine for single-night and multi-night campaigns; season-total visit
  demand carried across nights; Bernoulli clear-night weather model with
  seeded reproducibility; capacity warning for oversubscribed campaigns;
  unified console report and CSV trio output (blocks / per-target
  progress / nightly index).
- **`amase-precompute`** — parallel visibility precomputation with
  fingerprint-validated `.npz` cache, reusable across scheduling runs.
- **`amase-plot`** — plotting fully decoupled from scheduling:
  `campaign` (night-window utilization + completion bars, group-colored),
  `night` (altitude tracks + Gantt), `nights` (batch export), and
  `track` (single-target altitude/Moon-separation diagnostic for any
  site, with Moon illumination).
- Target list format with strict, row-numbered input validation
  (decimal/sexagesimal coordinates incl. fractional-minute
  normalization, `exp_time ≤ 3600`, `n_dither ∈ {1,3,9,27}`,
  `n_exposure ≥ 1`, optional `group` defaulting to `Untitle`).
- Constraints: altitude > 30°, Sun < −12° per slot, three-way Moon rule
  (illumination < 0.25 / below horizon / separation > 30°), 10-min
  per-block overhead; soft transit-preference weighting (`--alpha`),
  diversity (`--eps`) and completion (`--gamma`) bonuses.
- Documentation: `README.md` (user manual), `DESIGN.md` (algorithm &
  architecture), `example/demo.py` and `example/demo.ipynb`
  (end-to-end walkthroughs).
- Test suite: loader validation, single-night end-to-end feasibility,
  CSV trio round-trip (`pytest`).
