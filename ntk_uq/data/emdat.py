"""Build calibration initialization dates from the EM-DAT disaster database.

The paper evaluates on extreme-weather events from the EM-DAT International
Disaster Database. For each verified 2021 event, an initialization date is taken
a fixed number of days before event onset (a 3-day lookback in the paper), so
that forecasts are initialized during the event's development phase. Dates are
then deduplicated, since multiple concurrent disasters often share an
initialization date.

EM-DAT itself is not redistributed here (it has its own terms of use). Download
the public EM-DAT export from https://www.emdat.be/ and pass the .xlsx path to
``extract_precursor_dates``. The deduplicated date list used in the paper is
provided separately in ``data/calibration_dates.json``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

# Meteorologically relevant EM-DAT disaster types used in the paper.
METEOROLOGICAL_TYPES = ["Flood", "Storm", "Drought", "Extreme temperature"]

# WeatherBench2 ERA5 availability bounds.
WB2_START = datetime(1959, 1, 1)
WB2_END = datetime(2021, 12, 31)


def _event_onset(row) -> Optional[datetime]:
    """Parse an event onset date from an EM-DAT row (missing month/day -> 1)."""
    year = row.get("Start Year")
    if year is None or (isinstance(year, float) and year != year):  # NaN check
        return None
    month = row.get("Start Month")
    day = row.get("Start Day")
    month = int(month) if month == month else 1  # NaN -> 1
    day = int(day) if day == day else 1
    try:
        return datetime(int(year), month, day)
    except ValueError:
        return None


def extract_precursor_dates(
    emdat_path: str,
    year: int = 2021,
    buffer_days: int = 3,
    disaster_types: Optional[List[str]] = None,
) -> List[str]:
    """Extract deduplicated initialization dates from an EM-DAT export.

    Args:
        emdat_path: Path to the EM-DAT ``.xlsx`` export.
        year: Calendar year to filter events to (default 2021).
        buffer_days: Days before event onset for the initialization date.
        disaster_types: EM-DAT ``Disaster Type`` values to keep. Defaults to the
            meteorologically relevant subset used in the paper.

    Returns:
        Sorted list of unique initialization dates as ``"YYYY-MM-DD"`` strings,
        clipped to the WeatherBench2 ERA5 availability window.
    """
    import pandas as pd  # optional dependency (install via the ``data`` extra)

    if disaster_types is None:
        disaster_types = METEOROLOGICAL_TYPES

    df = pd.read_excel(emdat_path)
    df = df[df["Start Year"] == year]
    df = df[df["Disaster Type"].isin(disaster_types)]

    dates = set()
    for _, row in df.iterrows():
        onset = _event_onset(row)
        if onset is None:
            continue
        init = onset - timedelta(days=buffer_days)
        if WB2_START <= init <= WB2_END:
            dates.add(init.strftime("%Y-%m-%d"))

    return sorted(dates)
