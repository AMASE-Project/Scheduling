from amase_scheduling.output import (
    load_schedule_csvs,
    save_nights_csv,
    save_schedule_csv,
    save_targets_csv,
)


def test_csv_trio_roundtrip(tmp_path, night_result, targets):
    blocks_path = tmp_path / "plan.csv"
    targets_path = tmp_path / "plan_targets.csv"
    nights_path = tmp_path / "plan_nights.csv"
    save_schedule_csv(night_result, str(blocks_path))
    save_targets_csv(night_result, str(targets_path))
    save_nights_csv(night_result, str(nights_path))

    loaded = load_schedule_csvs(
        str(blocks_path),
        targets_path=str(targets_path),
        nights_path=str(nights_path),
        targets=targets,
    )

    assert loaded.start_date == night_result.start_date
    assert loaded.end_date == night_result.end_date
    assert len(loaded.nights) == len(night_result.nights)

    orig, new = night_result.nights[0], loaded.nights[0]
    assert new.clear == orig.clear
    assert len(new.blocks) == len(orig.blocks)
    assert {b.target_name for b in new.blocks} == {
        b.target_name for b in orig.blocks
    }

    done_orig = {p.name: p.done for p in night_result.progress}
    done_new = {p.name: p.done for p in loaded.progress}
    assert done_new == done_orig
