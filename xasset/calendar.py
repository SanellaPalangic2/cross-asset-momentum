"""NYSE trading calendar: the reference grid every asset is checked against."""
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=None)
def nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    """All NYSE full or half-day sessions in [start, end], as tz-naive dates.

    exchange_calendars covers ad-hoc closures (Hurricane Sandy 2012, the
    Bush and Carter national days of mourning) that a weekday-minus-holidays
    rule would miss.
    """
    cal = xcals.get_calendar("XNYS")
    first = max(pd.Timestamp(start), cal.first_session)
    last = min(pd.Timestamp(end), cal.last_session)
    sessions = cal.sessions_in_range(first, last)
    sessions = sessions.tz_localize(None) if sessions.tz else sessions
    return pd.DatetimeIndex(sessions, name="date").astype("datetime64[ns]")
