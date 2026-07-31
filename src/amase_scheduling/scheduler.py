from dataclasses import dataclass, field

import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import AltAz, SkyCoord, get_body
from astroplan import Observer

from amase_scheduling.observatory import NanshanObserver, night_window
from amase_scheduling.target import Target
from amase_scheduling.visibility import (
    SLOT_MINUTES,
    NightVisibility,
)
from amase_scheduling.cache import VisibilityCache, compute_night_visibility
from amase_scheduling.milp import build_milp, solve_milp
from amase_scheduling.weather import WeatherModel


@dataclass
class ScheduledBlock:
    target_name: str
    target_index: int
    target_coord: SkyCoord
    visit: int
    start_time: Time
    end_time: Time
    altitude: float
    azimuth: float
    moon_separation: float


@dataclass
class NightPlan:
    date: str
    clear: bool
    night_start: Time | None = None
    night_end: Time | None = None
    blocks: list[ScheduledBlock] = field(default_factory=list)
    unschedulable: list[str] = field(default_factory=list)
    unselected: list[str] = field(default_factory=list)

    @property
    def total_obs_time(self) -> float:
        if not self.blocks:
            return 0.0
        return sum(
            (b.end_time - b.start_time).to(u.min).value for b in self.blocks
        )

    @property
    def night_duration(self) -> float:
        if self.night_start is None or self.night_end is None:
            return 0.0
        return (self.night_end - self.night_start).to(u.min).value

    @property
    def n_scheduled_targets(self) -> int:
        return len({b.target_index for b in self.blocks})


@dataclass
class TargetProgress:
    name: str
    required: int
    done: int = 0
    nights_observed: int = 0
    obs_time_min: float = 0.0

    @property
    def fraction(self) -> float:
        return self.done / self.required if self.required > 0 else 1.0

    @property
    def completed(self) -> bool:
        return self.done >= self.required


@dataclass
class Schedule:
    start_date: str
    end_date: str
    clear_prob: float
    seed: int | None
    nights: list[NightPlan] = field(default_factory=list)
    progress: list[TargetProgress] = field(default_factory=list)
    capacity_warning: str | None = None

    @property
    def n_nights(self) -> int:
        return len(self.nights)

    @property
    def n_clear(self) -> int:
        return sum(1 for n in self.nights if n.clear)

    @property
    def n_completed(self) -> int:
        return sum(1 for p in self.progress if p.completed)

    @property
    def total_obs_time(self) -> float:
        return sum(p.obs_time_min for p in self.progress)

    @property
    def is_single_night(self) -> bool:
        return len(self.nights) == 1


class Scheduler:
    def __init__(self, observer: Observer | None = None):
        self.observer = observer or NanshanObserver()

    def schedule(
        self,
        targets: list[Target],
        start: str,
        end: str | None = None,
        clear_prob: float = 1.0,
        seed: int | None = None,
        eps: float = 1e-3,
        gamma: float = 0.01,
        alpha: float = 0.5,
        time_limit: int = 60,
        initial_remaining: np.ndarray | None = None,
        visibility_cache: VisibilityCache | None = None,
        verbose: bool = False,
    ) -> Schedule:
        t_start = Time(start, format="isot", scale="utc")
        t_end = Time(end, format="isot", scale="utc") if end else t_start
        if t_end < t_start:
            raise ValueError(f"end {end} is before start {start}")

        n_nights = int(np.round((t_end - t_start).to(u.day).value)) + 1

        if visibility_cache is not None:
            visibility_cache.validate(targets)
            if alpha > 0:
                sample = visibility_cache.get(t_start.isot[:10])
                if sample.quality is None:
                    raise ValueError(
                        "visibility cache has no transit-quality data "
                        "(built before quality support) — rebuild with amase-precompute, "
                        "or run with alpha=0."
                    )

        if initial_remaining is not None:
            remaining = np.asarray(initial_remaining, dtype=int).copy()
        else:
            remaining = np.array([t.n_exposure for t in targets], dtype=int)
        required = remaining.copy()

        result = Schedule(
            start_date=start,
            end_date=end or start,
            clear_prob=clear_prob,
            seed=seed,
            progress=[
                TargetProgress(name=t.name, required=int(r))
                for t, r in zip(targets, required)
            ],
            capacity_warning=(
                self._capacity_check(targets, n_nights, clear_prob)
                if n_nights > 1
                else None
            ),
        )

        weather = WeatherModel(clear_prob=clear_prob, seed=seed)

        for night_idx in range(n_nights):
            date = t_start + TimeDelta(night_idx * 86400, format="sec")
            date_str = date.isot[:10]

            if not weather.is_clear():
                result.nights.append(NightPlan(date=date_str, clear=False))
                if verbose:
                    print(f"{date_str}: cloudy")
                continue

            active_idx = np.where(remaining > 0)[0]
            if len(active_idx) == 0:
                result.nights.append(NightPlan(date=date_str, clear=True))
                if verbose:
                    print(f"{date_str}: all targets complete")
                continue

            active_targets = [targets[i] for i in active_idx]

            if visibility_cache is not None:
                full_vis = visibility_cache.get(date_str)
                night_vis = NightVisibility(
                    date_str,
                    full_vis.night_start,
                    full_vis.night_end,
                    full_vis.t0,
                    full_vis.valid_start[active_idx],
                    full_vis.quality[active_idx] if full_vis.quality is not None else None,
                )
            else:
                night_vis = self._night_visibility(active_targets, date_str)

            night = self._solve_night(
                active_targets,
                night_vis,
                k_remaining=remaining[active_idx],
                eps=eps,
                gamma=gamma,
                alpha=alpha,
                time_limit=time_limit,
            )

            for block in night.blocks:
                gi = active_idx[block.target_index]
                remaining[gi] -= 1
                prog = result.progress[gi]
                prog.done += 1
                prog.obs_time_min += (block.end_time - block.start_time).to(u.min).value
            for gi in {active_idx[b.target_index] for b in night.blocks}:
                result.progress[gi].nights_observed += 1

            result.nights.append(night)
            if verbose:
                print(
                    f"{date_str}: {len(night.blocks)} blocks, "
                    f"{night.total_obs_time:.0f} min"
                )

        return result

    def _night_visibility(
        self, targets: list[Target], date_str: str
    ) -> NightVisibility:
        return compute_night_visibility(self.observer, targets, date_str)

    def _solve_night(
        self,
        targets: list[Target],
        night_vis: NightVisibility,
        k_remaining: np.ndarray,
        eps: float,
        gamma: float,
        alpha: float,
        time_limit: int,
    ) -> NightPlan:
        date_str = night_vis.date
        night_start = night_vis.night_start
        night_end = night_vis.night_end
        t0 = night_vis.t0
        valid_start = night_vis.valid_start
        quality = night_vis.quality

        N = len(targets)
        B_i = np.array(
            [max(1, int(np.ceil(t.block_duration_sec / (SLOT_MINUTES * 60)))) for t in targets],
            dtype=int,
        )
        K_i = np.asarray(k_remaining, dtype=int)
        w_i = np.array([t.priority for t in targets], dtype=float)

        unschedulable = []
        schedulable_indices = []
        for i in range(N):
            if not np.any(valid_start[i]):
                unschedulable.append(targets[i].name)
            else:
                schedulable_indices.append(i)

        if not schedulable_indices:
            return NightPlan(
                date=date_str,
                clear=True,
                night_start=night_start,
                night_end=night_end,
                unschedulable=unschedulable,
            )

        idx_map = {}
        valid_start_sub = np.zeros((len(schedulable_indices), valid_start.shape[1]), dtype=bool)
        quality_sub = (
            np.zeros((len(schedulable_indices), valid_start.shape[1]), dtype=np.float32)
            if quality is not None
            else None
        )
        for new_i, old_i in enumerate(schedulable_indices):
            valid_start_sub[new_i] = valid_start[old_i]
            if quality_sub is not None:
                quality_sub[new_i] = quality[old_i]
            idx_map[new_i] = old_i

        B_sub = B_i[schedulable_indices]
        K_sub = K_i[schedulable_indices]
        w_sub = w_i[schedulable_indices]

        prob, var_map, solver = build_milp(
            valid_start_sub, B_sub, K_sub, w_sub, eps=eps, gamma=gamma,
            time_limit=time_limit, quality=quality_sub, alpha=alpha,
        )
        raw_schedule = solve_milp(prob, var_map, solver)

        scheduled_names = set()
        blocks = []
        visit_counter: dict[int, int] = {}
        for new_i, k in raw_schedule:
            old_i = idx_map[new_i]
            target = targets[old_i]
            B = int(B_sub[new_i])
            visit_counter[old_i] = visit_counter.get(old_i, 0) + 1

            start_time = t0[k]
            end_time = t0[k] + TimeDelta(B * SLOT_MINUTES * 60, format="sec")
            mid_time = t0[k] + TimeDelta(B * SLOT_MINUTES * 30, format="sec")

            frame = AltAz(obstime=mid_time, location=self.observer.location)
            altaz = target.coord.transform_to(frame)
            moon = get_body("moon", mid_time, location=self.observer.location)
            moon_altaz = moon.transform_to(frame)
            moon_sep = float(altaz.separation(moon_altaz).deg)

            blocks.append(
                ScheduledBlock(
                    target_name=target.name,
                    target_index=old_i,
                    target_coord=target.coord,
                    visit=visit_counter[old_i],
                    start_time=start_time,
                    end_time=end_time,
                    altitude=float(altaz.alt.deg),
                    azimuth=float(altaz.az.deg),
                    moon_separation=moon_sep,
                )
            )
            scheduled_names.add(target.name)

        unselected = [
            t.name
            for t in targets
            if t.name not in scheduled_names and t.name not in unschedulable
        ]

        return NightPlan(
            date=date_str,
            clear=True,
            night_start=night_start,
            night_end=night_end,
            blocks=blocks,
            unschedulable=unschedulable,
            unselected=unselected,
        )

    def _capacity_check(
        self, targets: list[Target], n_nights: int, clear_prob: float
    ) -> str | None:
        demand_min = sum(t.total_time_sec / 60 for t in targets)
        sample_dates = [
            Time("2026-01-01", format="isot"),
            Time("2026-04-01", format="isot"),
            Time("2026-07-01", format="isot"),
            Time("2026-10-01", format="isot"),
        ]
        night_hours = []
        for d in sample_dates:
            s, e = night_window(self.observer, d)
            night_hours.append((e - s).to(u.hour).value)
        mean_night_min = float(np.mean(night_hours)) * 60
        expected_available = n_nights * mean_night_min * clear_prob
        if demand_min > expected_available:
            return (
                f"CAPACITY WARNING: total demand {demand_min:.0f} min exceeds "
                f"expected available {expected_available:.0f} min "
                f"({n_nights} nights x {mean_night_min:.0f} min x {clear_prob:.0%} clear). "
                f"Some targets will not complete."
            )
        return None
