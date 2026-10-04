from pathlib import Path

import pytest

from finbench.data import load_examples

FIXTURE = Path(__file__).parent / "fixtures" / "finqa_sample.json"


@pytest.fixture
def examples():
    """Six real FinQA test items (MIT licensed, github.com/czyssrs/FinQA):
    1-step, 2-step, 5-step with constants, table_average, greater, percent literal."""
    return {e.id: e for e in load_examples(FIXTURE)}
