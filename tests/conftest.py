from pathlib import Path

import pytest

from amase_scheduling import Scheduler, load_targets

REPO = Path(__file__).resolve().parent.parent
EXAMPLE_TARGETS = REPO / "example" / "targets.csv"
TEST_DATE = "2027-05-01"


@pytest.fixture(scope="session")
def targets():
    return load_targets(str(EXAMPLE_TARGETS))


@pytest.fixture(scope="session")
def night_result(targets):
    return Scheduler().schedule(targets, TEST_DATE)
