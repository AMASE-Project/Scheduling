import argparse
import sys
import time

from amase_scheduling.cache import VisibilityCache
from amase_scheduling.target import load_targets


def main():
    from amase_scheduling._warnings import suppress_future_date_warnings
    suppress_future_date_warnings()

    parser = argparse.ArgumentParser(
        description="AMASE-P visibility precomputer: compute per-night "
        "visibility in parallel and save to disk for reuse by amase-schedule.",
    )
    parser.add_argument("targets", help="Path to target list (CSV)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output cache file (e.g. vis_cache.npz)")
    args = parser.parse_args()

    try:
        targets = load_targets(args.targets)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading targets: {e}", file=sys.stderr)
        sys.exit(1)
    if not targets:
        print(f"Error: No targets loaded from {args.targets}", file=sys.stderr)
        sys.exit(1)

    print(f"Precomputing visibility: {len(targets)} targets, "
          f"{args.start} .. {args.end}, workers={args.workers}")
    t0 = time.perf_counter()
    try:
        cache = VisibilityCache.build(
            targets, args.start, args.end, n_workers=args.workers
        )
    except ValueError as e:
        print(f"Error precomputing: {e}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    cache.save(args.output)
    print(f"Saved {len(cache)} nights to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
