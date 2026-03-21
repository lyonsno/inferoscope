"""Shared format validators for inferoscope artifact contracts."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import re


_RFC3339_DATETIME_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"[Tt]"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"(?P<fraction>\.\d+)?"
    r"(?:(?P<utc_offset>[Zz])|(?P<offset_sign>[+-])(?P<offset_hour>\d{2}):(?P<offset_minute>\d{2}))$"
)

_KNOWN_UTC_LEAP_SECOND_DATES = frozenset(
    {
        date(1972, 6, 30),
        date(1972, 12, 31),
        date(1973, 12, 31),
        date(1974, 12, 31),
        date(1975, 12, 31),
        date(1976, 12, 31),
        date(1977, 12, 31),
        date(1978, 12, 31),
        date(1979, 12, 31),
        date(1981, 6, 30),
        date(1982, 6, 30),
        date(1983, 6, 30),
        date(1985, 6, 30),
        date(1987, 12, 31),
        date(1989, 12, 31),
        date(1990, 12, 31),
        date(1992, 6, 30),
        date(1993, 6, 30),
        date(1994, 6, 30),
        date(1995, 12, 31),
        date(1997, 6, 30),
        date(1998, 12, 31),
        date(2005, 12, 31),
        date(2008, 12, 31),
        date(2012, 6, 30),
        date(2015, 6, 30),
        date(2016, 12, 31),
    }
)


def _match_int(match: re.Match[str], group_name: str) -> int:
    return int(match.group(group_name))


def _fraction_to_microseconds(fraction: str | None) -> int:
    if fraction is None:
        return 0

    digits = fraction[1:]
    return int((digits[:6]).ljust(6, "0"))


def _match_timezone(match: re.Match[str]) -> timezone | None:
    if match.group("utc_offset") is not None:
        return timezone.utc

    offset_hour = _match_int(match, "offset_hour")
    offset_minute = _match_int(match, "offset_minute")
    if offset_hour > 23 or offset_minute > 59:
        return None

    offset_delta = timedelta(hours=offset_hour, minutes=offset_minute)
    if match.group("offset_sign") == "-":
        offset_delta = -offset_delta

    return timezone(offset_delta)


def _build_matched_datetime(match: re.Match[str], *, second: int) -> datetime | None:
    tzinfo = _match_timezone(match)
    if tzinfo is None:
        return None

    try:
        return datetime(
            _match_int(match, "year"),
            _match_int(match, "month"),
            _match_int(match, "day"),
            _match_int(match, "hour"),
            _match_int(match, "minute"),
            second,
            _fraction_to_microseconds(match.group("fraction")),
            tzinfo=tzinfo,
        )
    except ValueError:
        return None


def _is_valid_leap_second(match: re.Match[str]) -> bool:
    leap_second_moment = _build_matched_datetime(match, second=59)
    if leap_second_moment is None:
        return False

    utc_moment = leap_second_moment.astimezone(timezone.utc)
    return (
        utc_moment.date() in _KNOWN_UTC_LEAP_SECOND_DATES
        and utc_moment.hour == 23
        and utc_moment.minute == 59
        and utc_moment.second == 59
    )


def is_rfc3339_datetime(value: str) -> bool:
    """Return True when ``value`` is an RFC3339 date-time string."""

    match = _RFC3339_DATETIME_RE.match(value)
    if match is None:
        return False

    second = _match_int(match, "second")
    if second > 60:
        return False

    if second == 60:
        return _is_valid_leap_second(match)

    return _build_matched_datetime(match, second=second) is not None
