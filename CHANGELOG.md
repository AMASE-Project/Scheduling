# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
