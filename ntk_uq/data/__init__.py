"""Dataset utilities: EM-DAT initialization dates and the released date lists.

The two date lists used in the paper are bundled with the package and loaded via
``importlib.resources`` so they are available after a plain ``pip install``:

    load_calibration_dates()  -> the 100 calibration initialization dates (paper n).
    load_precursor_pool()     -> the 208-date EM-DAT precursor pool (3-day lookback).
"""

import json
from importlib import resources
from typing import List

from .emdat import extract_precursor_dates, METEOROLOGICAL_TYPES


def _load(name: str) -> List[str]:
    with resources.files(__name__).joinpath(name).open(encoding="utf-8") as f:
        return json.load(f)


def load_calibration_dates() -> List[str]:
    """Return the 100 EM-DAT calibration initialization dates used in the paper."""
    return _load("calibration_dates.json")


def load_precursor_pool() -> List[str]:
    """Return the full 208-date EM-DAT precursor pool (3-day lookback)."""
    return _load("raw.json")


__all__ = [
    "extract_precursor_dates",
    "METEOROLOGICAL_TYPES",
    "load_calibration_dates",
    "load_precursor_pool",
]
