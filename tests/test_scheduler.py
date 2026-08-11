OVERHEAD_MIN = 10.0


def test_single_night_blocks_feasible(night_result):
    assert len(night_result.nights) == 1
    night = night_result.nights[0]
    assert night.clear
    assert night.blocks, "expected at least one scheduled block"

    ordered = sorted(night.blocks, key=lambda b: b.start_time)
    for b in ordered:
        assert b.start_time >= night.night_start
        assert b.end_time <= night.night_end
    for a, b in zip(ordered, ordered[1:]):
        gap_min = (b.start_time - a.end_time).to_value("min")
        if a.target_name == b.target_name:
            assert gap_min >= -1e-6
        else:
            assert gap_min >= OVERHEAD_MIN - 1e-6


def test_exposure_caps_respected(night_result, targets):
    required = {t.name: t.n_exposures for t in targets}
    exposures = {}
    for night in night_result.nights:
        for b in night.blocks:
            exposures[b.target_name] = exposures.get(b.target_name, 0) + 1
    assert exposures, "expected at least one exposure"
    for name, n in exposures.items():
        assert n <= required[name]
