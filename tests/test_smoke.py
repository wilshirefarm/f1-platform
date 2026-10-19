import importlib

import pytest

SUBPACKAGES = ["events", "replay", "features", "predict", "sim", "dq", "finetune"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"f1platform.{name}")
    assert module.__doc__
