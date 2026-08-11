import pytest

from amase_scheduling.target import load_targets

HEADER = "name,ra,dec,priority,exp_time,n_dither,n_set\n"
ROW = "M31,10.6847,41.2691,1,300,3,2\n"


def _write(tmp_path, text):
    p = tmp_path / "targets.csv"
    p.write_text(text)
    return str(p)


def test_minimal_valid_loads_with_default_group(tmp_path):
    (t,) = load_targets(_write(tmp_path, HEADER + ROW))
    assert t.name == "M31"
    assert t.group == "Untitle"
    assert t.block_duration_sec == 300.0
    assert t.n_exposures == 6


def test_sexagesimal_and_fractional_minute(tmp_path):
    (t,) = load_targets(
        _write(tmp_path, HEADER + "IC434,05h40.9m0.0s,-01d30m00s,1,600,3,2\n")
    )
    assert t.coord.ra.deg == pytest.approx(85.225, abs=1e-3)
    assert t.coord.dec.deg == pytest.approx(-1.5, abs=1e-3)


@pytest.mark.parametrize(
    "row, fragment",
    [
        ("T,foo,47.3,1,600,3,2\n", "ra format"),
        ("T,184.7,,1,600,3,2\n", "'dec'"),
        ("T,184.7,47.3,1,3601,3,2\n", "exp_time"),
        ("T,184.7,47.3,1,600,5,2\n", "n_dither"),
        ("T,184.7,47.3,1,600,3,0\n", "n_set"),
        (",184.7,47.3,1,600,3,2\n", "'name'"),
    ],
)
def test_invalid_rows_rejected(tmp_path, row, fragment):
    with pytest.raises(ValueError, match=fragment):
        load_targets(_write(tmp_path, HEADER + row))


def test_missing_column_rejected(tmp_path):
    bad = "name,ra,dec,priority,exp_time,n_dither\nT,184.7,47.3,1,600,3\n"
    with pytest.raises(ValueError, match="n_set"):
        load_targets(_write(tmp_path, bad))
