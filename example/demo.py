"""AMASE-P scheduling complete usage demo.

Workflow demonstrated:
  1. load target list
  2. precompute visibility cache in parallel (once)
  3. single-night schedule (reuses the cache)
  4. multi-night campaign 2027-04-01 .. 2027-04-15 (reuses the same cache)
  5. observing log (text + CSV) and all figures

Results are written to example/outputs/.
Run from anywhere:  python3 example/demo.py
"""

from pathlib import Path

from amase_scheduling import (
    Scheduler,
    VisibilityCache,
    format_report,
    load_targets,
    plot_campaign_figure,
    plot_night_figure,
    save_all_night_figures,
    save_nights_csv,
    save_schedule_csv,
    save_targets_csv,
)

REPO = Path(__file__).resolve().parent.parent
TARGETS = REPO / "example" / "targets.csv"
OUTDIR = REPO / "example" / "outputs"

START, END = "2027-04-01", "2027-04-15"
CLEAR_PROB = 0.5
SEED = 2027
WORKERS = 8


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets(str(TARGETS))
    print(f"Loaded {len(targets)} targets from {TARGETS.name}")

    print(f"\n[1/4] Precomputing visibility {START} .. {END} ({WORKERS} workers) ...")
    cache = VisibilityCache.build(targets, START, END, n_workers=WORKERS)
    cache.save(str(OUTDIR / "vis_cache.npz"))
    print(f"      cached {len(cache)} nights -> {OUTDIR / 'vis_cache.npz'}")

    scheduler = Scheduler()

    print(f"\n[2/4] Single-night schedule: {START}")
    night_result = scheduler.schedule(
        targets, START, visibility_cache=cache,
    )
    print(format_report(night_result))
    plot_night_figure(
        night_result.nights[0], targets, scheduler.observer,
        str(OUTDIR / f"night_{START}.png"),
    )

    print(f"\n[3/4] Campaign simulation {START} .. {END} "
          f"(clear_prob={CLEAR_PROB}, seed={SEED})")
    result = scheduler.schedule(
        targets, START, END,
        clear_prob=CLEAR_PROB, seed=SEED,
        visibility_cache=cache, verbose=True,
    )

    print("\n[4/4] Writing observing log and figures ...")
    report = format_report(result)
    print(report)
    (OUTDIR / "observing_log.txt").write_text(report + "\n")

    save_schedule_csv(result, str(OUTDIR / "observing_log.csv"))
    save_targets_csv(result, str(OUTDIR / "observing_log_targets.csv"))
    save_nights_csv(result, str(OUTDIR / "observing_log_nights.csv"))
    plot_campaign_figure(result, str(OUTDIR / "campaign.png"),
                         targets=targets, observer=scheduler.observer)
    night_paths = save_all_night_figures(
        result, targets, scheduler.observer, str(OUTDIR / "nights")
    )

    print(f"\nDone. Results in {OUTDIR}:")
    print(f"  observing_log.txt / observing_log.csv / observing_log_targets.csv / observing_log_nights.csv")
    print(f"  campaign.png, night_{START}.png, nights/ ({len(night_paths)} figures)")
    print(f"  vis_cache.npz")


if __name__ == "__main__":
    main()
