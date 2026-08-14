"""Tests for the /api/targets/parse and /api/targets/example endpoints."""

from fastapi.testclient import TestClient

from amase_scheduling.web.main import app

client = TestClient(app)


VALID_CSV = (
    "name,ra,dec,priority,exp_time,n_dither,n_set,group,comments\n"
    "NGC4258,184.739583,47.303972,1,3600,9,1,Zongnan Li,\n"
    "M51,202.462917,47.409444,1,600,27,1,Zhijie Qu,\n"
)


def test_parse_valid_csv():
    r = client.post("/api/targets/parse", json={"csv": VALID_CSV})
    assert r.status_code == 200
    data = r.json()
    assert data["n_rows"] == 2
    assert data["n_errors"] == 0
    assert data["errors"] == []
    assert len(data["rows"]) == 2

    first = data["rows"][0]
    assert first["name"] == "NGC4258"
    assert first["ra"] == "184.739583"
    assert first["priority"] == 1.0
    assert first["exp_time"] == 3600.0
    assert first["n_dither"] == 9
    assert first["n_set"] == 1
    assert first["group"] == "Zongnan Li"


def test_parse_missing_required_column():
    bad = "name,ra,dec,priority,exp_time,n_dither\nfoo,1,2,1,60,1\n"
    r = client.post("/api/targets/parse", json={"csv": bad})
    assert r.status_code == 400
    assert "missing required columns" in r.json()["detail"]


def test_parse_empty_csv():
    r = client.post("/api/targets/parse", json={"csv": ""})
    assert r.status_code == 400


def test_parse_invalid_rows_report_line_numbers_and_reasons():
    # Each of these rows violates a different rule:
    #   2: exp_time out of range (> 3600)
    #   3: n_dither not in {1,3,9,27}
    #   4: n_set < 1
    #   5: priority not a number
    #   6: missing required field (dec empty)
    #   7: invalid coordinates
    csv_text = (
        "name,ra,dec,priority,exp_time,n_dither,n_set\n"
        "A,10.0,20.0,1,5000,9,1\n"
        "B,10.0,20.0,1,60,2,1\n"
        "C,10.0,20.0,1,60,9,0\n"
        "D,10.0,20.0,abc,60,9,1\n"
        "E,10.0,,1,60,9,1\n"
        "F,25h00m00s,20.0,1,60,9,1\n"
    )
    r = client.post("/api/targets/parse", json={"csv": csv_text})
    assert r.status_code == 200
    data = r.json()
    assert data["n_errors"] == 6
    # All data lines are returned as rows (raw values for invalid lines),
    # with 1-based data-line numbers in errors.
    assert data["n_rows"] == 6
    assert len(data["rows"]) == 6
    assert data["rows"][0]["name"] == "A"
    assert data["rows"][0]["exp_time"] == "5000"

    errors = {e["line"]: e["message"] for e in data["errors"]}
    assert errors[1].startswith("exp_time must be in")
    assert "n_dither must be one of" in errors[2]
    assert "n_set must be >= 1" in errors[3]
    assert "invalid priority" in errors[4]
    assert "missing value for required field 'dec'" in errors[5]
    assert "invalid coordinates" in errors[6]


def test_example_endpoint_returns_csv_text():
    r = client.get("/api/targets/example")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    body = r.text
    assert "name,ra,dec,priority,exp_time,n_dither,n_set" in body
    assert "NGC4258" in body
