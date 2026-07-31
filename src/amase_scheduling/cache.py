import multiprocessing as mp

import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astroplan import Observer

from amase_scheduling.observatory import NanshanObserver, night_window
from amase_scheduling.target import Target
from amase_scheduling.visibility import (
    SLOT_MINUTES,
    NightVisibility,
    compute_quality,
    compute_valid_starts,
    compute_visibility,
)


def block_slots_of(targets: list[Target]) -> np.ndarray:
    return np.array(
        [max(1, int(np.ceil(t.block_duration_sec / (SLOT_MINUTES * 60)))) for t in targets],
        dtype=int,
    )


def compute_night_visibility(
    observer: Observer,
    targets: list[Target],
    date_str: str,
    block_slots: np.ndarray | None = None,
) -> NightVisibility:
    """The expensive astronomy for one night: night window, visibility
    matrix, valid starts. Independent of weather and scheduling state."""
    if block_slots is None:
        block_slots = block_slots_of(targets)
    date = Time(date_str, format="isot", scale="utc")
    night_start, night_end = night_window(observer, date)
    visible, t0, alt = compute_visibility(observer, targets, night_start, night_end)
    valid_start = compute_valid_starts(visible, block_slots)
    quality = compute_quality(alt, visible, valid_start, block_slots)
    return NightVisibility(date_str, night_start, night_end, t0, valid_start, quality)


def _worker_one_night(args) -> NightVisibility:
    date_str, targets, block_slots = args
    observer = NanshanObserver()
    return compute_night_visibility(observer, targets, date_str, block_slots)


class VisibilityCache:
    """In-memory precomputed visibility for a fixed target list over a set
    of nights. Can be shared across multiple schedule() runs (e.g. Monte
    Carlo weather realizations)."""

    def __init__(self, target_names: list[str]):
        self.target_names = list(target_names)
        self.nights: dict[str, NightVisibility] = {}

    @classmethod
    def build(
        cls,
        targets: list[Target],
        start: str,
        end: str,
        n_workers: int = 1,
    ) -> "VisibilityCache":
        t_start = Time(start, format="isot", scale="utc")
        t_end = Time(end, format="isot", scale="utc")
        if t_end < t_start:
            raise ValueError(f"end {end} is before start {start}")
        n_nights = int(np.round((t_end - t_start).to(u.day).value)) + 1
        dates = [
            (t_start + TimeDelta(i * 86400, format="sec")).isot[:10]
            for i in range(n_nights)
        ]

        cache = cls([t.name for t in targets])
        block_slots = block_slots_of(targets)

        if n_workers > 1 and n_nights > 1:
            args = [(d, targets, block_slots) for d in dates]
            with mp.Pool(n_workers) as pool:
                results = pool.map(_worker_one_night, args)
            for nv in results:
                cache.nights[nv.date] = nv
        else:
            observer = NanshanObserver()
            for d in dates:
                nv = compute_night_visibility(observer, targets, d, block_slots)
                cache.nights[nv.date] = nv
        return cache

    def validate(self, targets: list[Target]) -> None:
        names = [t.name for t in targets]
        if names != self.target_names:
            raise ValueError(
                "VisibilityCache was built for a different target list "
                f"({len(self.target_names)} targets, first: "
                f"{self.target_names[:3]}) — rebuild the cache."
            )

    def get(self, date_str: str) -> NightVisibility:
        try:
            return self.nights[date_str]
        except KeyError:
            raise ValueError(
                f"VisibilityCache has no data for {date_str} "
                f"(covers {min(self.nights)} .. {max(self.nights)})"
            )

    def __len__(self) -> int:
        return len(self.nights)

    def save(self, path: str) -> None:
        dates = sorted(self.nights.keys())
        arrays: dict[str, np.ndarray] = {
            "target_names": np.array(self.target_names),
            "dates": np.array(dates),
        }
        for d in dates:
            nv = self.nights[d]
            arrays[f"vs::{d}"] = nv.valid_start
            arrays[f"t0::{d}"] = nv.t0.jd
            arrays[f"win::{d}"] = np.array([nv.night_start.jd, nv.night_end.jd])
            if nv.quality is not None:
                arrays[f"q::{d}"] = nv.quality
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "VisibilityCache":
        data = np.load(path, allow_pickle=False)
        target_names = [str(n) for n in data["target_names"]]
        cache = cls(target_names)
        for d in data["dates"]:
            date_str = str(d)
            win = data[f"win::{date_str}"]
            t0 = Time(data[f"t0::{date_str}"], format="jd", scale="utc")
            quality = data[f"q::{date_str}"] if f"q::{date_str}" in data else None
            cache.nights[date_str] = NightVisibility(
                date_str,
                Time(win[0], format="jd", scale="utc"),
                Time(win[1], format="jd", scale="utc"),
                t0,
                data[f"vs::{date_str}"],
                quality,
            )
        return cache
