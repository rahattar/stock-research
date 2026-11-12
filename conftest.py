"""Skip the one inherited test that needs the FX data of the forex-research repo."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    if (Path(__file__).parent / "fresh" / "download").exists():
        return
    skip = pytest.mark.skip(reason="needs the FX data of the forex-research repo")
    for item in items:
        if item.name == "test_task12_full_outputs_reproduce_unchanged":
            item.add_marker(skip)
