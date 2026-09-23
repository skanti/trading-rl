"""Refresh the shared US-equity market-calendar snapshot."""

import argparse
from datetime import date, datetime
from pathlib import Path

from ..market_data.calendar import EASTERN
from ..market_data.session_calendar import DEFAULT_CALENDAR_PATH, load_calendar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2022-01-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_CALENDAR_PATH)
    args = parser.parse_args()
    snapshot = load_calendar(
        date.fromisoformat(args.since), datetime.now(EASTERN).date(),
        path=args.output, refresh=True,
    )
    print(f"Refreshed market calendar: {args.output}; {len(snapshot.sessions):,} sessions since {args.since}")


if __name__ == "__main__":
    main()
