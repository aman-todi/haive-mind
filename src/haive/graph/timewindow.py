"""Time windows in history questions ("in the last 2 weeks", "since August", "last 20 commits"),
parsed in code and turned into ``git log --since`` / ``-n``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
WORD_NUMS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "few": 3, "couple of": 2}  # fmt: skip
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}  # fmt: skip
MONTH_RE = "|".join([*MONTHS, *(m[:3] for m in MONTHS if m != "may")])
NUM = r"(\d+|" + "|".join(sorted(WORD_NUMS, key=len, reverse=True)) + r")"

# Default for open questions like "what changed recently?": the last two weeks, or the newest
# commits if that window is (nearly) empty.
DEFAULT_DAYS = 14
DEFAULT_MIN_COMMITS = 3


@dataclass
class Window:
    since: date | None = None  # commits on or after this day (UTC)
    commits: int | None = None  # or: the newest N commits
    label: str = ""  # how the window reads back to the user

    @property
    def git_since(self) -> str | None:
        return self.since.isoformat() if self.since else None


def _num(s: str) -> int:
    s = s.lower()
    return int(s) if s.isdigit() else WORD_NUMS.get(s, 1)


def _month(name: str) -> int:
    name = name.lower()
    return MONTHS.get(name) or next(i for m, i in MONTHS.items() if m.startswith(name))


def parse_window(text: str, today: date | None = None) -> Window | None:
    """The first time window stated in ``text``, or None."""
    today = today or datetime.now(UTC).date()
    s = text.lower()

    m = re.search(rf"\b(?:last|latest|past|newest|recent)\s+{NUM}\s+commits?\b", s)
    if m:
        n = _num(m.group(1))
        return Window(commits=n, label=f"the last {n} commits")

    m = re.search(
        rf"\b(?:in\s+the\s+|over\s+the\s+|within\s+the\s+)?(?:last|past)\s+{NUM}\s+(day|week|month|year)s?\b",
        s,
    )
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        return Window(
            since=today - timedelta(days=n * UNIT_DAYS[unit]), label=f"the last {n} {unit}{'s' * (n != 1)}"
        )

    m = re.search(r"\b(?:last|past|this)\s+(day|week|month|year)\b", s)
    if m:
        unit = m.group(1)
        return Window(since=today - timedelta(days=UNIT_DAYS[unit]), label=f"the last {unit}")

    if re.search(r"\btoday\b", s):
        return Window(since=today, label="today")
    if re.search(r"\byesterday\b", s):
        return Window(since=today - timedelta(days=1), label="since yesterday")

    m = re.search(r"\bsince\s+(\d{4}-\d{2}-\d{2})\b", s)
    if m:
        return Window(since=date.fromisoformat(m.group(1)), label=f"since {m.group(1)}")

    m = re.search(rf"\b(?:since|from|in)\s+({MONTH_RE})\b\.?(?:\s+(\d{{4}}))?", s)
    if m:
        month = _month(m.group(1))
        year = int(m.group(2)) if m.group(2) else today.year - (month > today.month)
        start = date(year, month, 1)
        return Window(since=start, label=f"since {start.strftime('%B %Y')}")
    return None


def default_window(today: date | None = None) -> Window:
    today = today or datetime.now(UTC).date()
    return Window(since=today - timedelta(days=DEFAULT_DAYS), label=f"the last {DEFAULT_DAYS // 7} weeks")
