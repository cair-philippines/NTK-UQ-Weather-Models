"""Load the EM-DAT initialization dates, or rebuild them from an EM-DAT export.

The repository ships two date lists under ``data/``:

    calibration_dates.json  - the 100 initialization dates used to build the
                              calibration set in the paper (n = 100).
    raw.json                - the full 208-date EM-DAT precursor pool (3-day
                              lookback) the calibration set is drawn from.

This script loads the released list and, optionally, re-derives the precursor
pool from a raw EM-DAT .xlsx export to show the extraction is reproducible.

Run:
    python examples/03_emdat_dates.py
    python examples/03_emdat_dates.py --emdat path/to/public_emdat.xlsx
"""

import argparse

from ntk_uq.data import load_calibration_dates, load_precursor_pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emdat", default=None,
                        help="Optional path to a raw EM-DAT .xlsx export to re-extract from.")
    args = parser.parse_args()

    calib = load_calibration_dates()
    pool = load_precursor_pool()

    print(f"Calibration set (paper n): {len(calib)} dates")
    print(f"  first 5: {calib[:5]}")
    print(f"EM-DAT precursor pool    : {len(pool)} dates")
    print(f"Calibration subset of pool: {set(calib).issubset(set(pool))}")

    if args.emdat:
        from ntk_uq.data import extract_precursor_dates

        rebuilt = extract_precursor_dates(args.emdat, year=2021, buffer_days=3)
        print(f"\nRe-extracted from {args.emdat}: {len(rebuilt)} precursor dates")
        print(f"  first 5: {rebuilt[:5]}")


if __name__ == "__main__":
    main()
