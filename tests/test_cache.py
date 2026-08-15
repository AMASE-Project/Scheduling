"""Tests for the VisibilityCache exposure-time fingerprint: save/load
round-trip of ``block_slots`` and the ``validate`` name + fingerprint checks."""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from amase_scheduling.cache import VisibilityCache, block_slots_of
from amase_scheduling.target import Target
from amase_scheduling.visibility import NightVisibility

DATE = "2027-04-01"


def _make_target(name: str, exp_time: float) -> Target:
    return Target(
        name=name,
        coord=SkyCoord(0.0, 0.0, unit="deg"),
        priority=1.0,
        exp_time=exp_time,
        n_dither=1,
        n_set=1,
    )


def _make_cache(targets: list[Target]) -> VisibilityCache:
    """Build a tiny single-night cache in memory (no astronomy needed)."""
    cache = VisibilityCache(
        [t.name for t in targets],
        block_slots=block_slots_of(targets),
    )
    night_start = Time(f"{DATE}T17:00:00", format="isot", scale="utc")
    night_end = Time(f"{DATE}T23:00:00", format="isot", scale="utc")
    t0 = Time(f"{DATE}T18:00:00", format="isot", scale="utc")
    n_slots = 12
    valid_start = np.zeros((len(targets), n_slots), dtype=bool)
    valid_start[:, :3] = True
    quality = np.ones((len(targets), n_slots), dtype=np.float32)
    cache.nights[DATE] = NightVisibility(
        DATE, night_start, night_end, t0, valid_start, quality
    )
    return cache


def test_build_attaches_block_slots():
    targets = [_make_target("A", 600), _make_target("B", 300)]
    cache = _make_cache(targets)
    assert list(cache.block_slots) == [2, 1]


def test_save_load_roundtrip_preserves_block_slots(tmp_path):
    targets = [_make_target("A", 600), _make_target("B", 300)]
    cache = _make_cache(targets)
    p = tmp_path / "cache.npz"
    cache.save(str(p))

    loaded = VisibilityCache.load(str(p))
    assert loaded.target_names == ["A", "B"]
    assert list(loaded.block_slots) == [2, 1]
    assert DATE in loaded.nights
    loaded.validate(targets)  # matching names + fingerprint -> no raise


def test_load_bytes_matches_load_path(tmp_path):
    targets = [_make_target("A", 600), _make_target("B", 300)]
    cache = _make_cache(targets)
    p = tmp_path / "cache.npz"
    cache.save(str(p))

    from_path = VisibilityCache.load(str(p))
    from_bytes = VisibilityCache.load_bytes(p.read_bytes())

    assert from_bytes.target_names == from_path.target_names == ["A", "B"]
    assert len(from_bytes) == len(from_path) == 1
    assert list(from_bytes.block_slots) == list(from_path.block_slots) == [2, 1]
    nv_path = from_path.nights[DATE]
    nv_bytes = from_bytes.nights[DATE]
    assert np.array_equal(nv_bytes.valid_start, nv_path.valid_start)
    assert np.array_equal(nv_bytes.quality, nv_path.quality)


def test_validate_raises_on_exposure_time_change():
    targets = [_make_target("A", 600), _make_target("B", 300)]
    cache = _make_cache(targets)
    changed = [_make_target("A", 300), _make_target("B", 300)]
    with pytest.raises(ValueError, match="block-slot fingerprint"):
        cache.validate(changed)


def test_validate_raises_on_name_change():
    cache = _make_cache([_make_target("A", 600)])
    with pytest.raises(ValueError, match="different target list"):
        cache.validate([_make_target("C", 600)])


def test_old_format_cache_warns_but_passes(tmp_path):
    targets = [_make_target("A", 600)]
    cache = _make_cache(targets)
    p = tmp_path / "cache.npz"
    cache.save(str(p))

    # Strip the fingerprint to simulate a pre-fingerprint cache file.
    with np.load(str(p), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "block_slots"}
    old = tmp_path / "old.npz"
    np.savez_compressed(str(old), **arrays)

    loaded = VisibilityCache.load(str(old))
    assert loaded.block_slots is None
    with pytest.warns(UserWarning, match="no exposure-time fingerprint"):
        loaded.validate(targets)  # names match -> passes with a warning
