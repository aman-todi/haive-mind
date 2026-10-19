from datetime import date

import pytest

from haive.graph.timewindow import default_window, parse_window

TODAY = date(2026, 10, 19)


@pytest.mark.parametrize(
    ("text", "since", "commits", "label"),
    [
        ("what changed in the last 2 weeks?", date(2026, 10, 5), None, "the last 2 weeks"),
        ("changes over the past three months", date(2026, 7, 21), None, "the last 3 months"),
        ("what did I do in the last few days", date(2026, 10, 16), None, "the last 3 days"),
        ("show the last 20 commits", None, 20, "the last 20 commits"),
        ("latest five commits in weatherpruf", None, 5, "the last 5 commits"),
        ("what changed this week?", date(2026, 10, 12), None, "the last week"),
        ("anything new today", TODAY, None, "today"),
        ("since 2026-09-01, what changed?", date(2026, 9, 1), None, "since 2026-09-01"),
        ("what changed since August", date(2026, 8, 1), None, "since August 2026"),
        ("what changed in sep 2025", date(2025, 9, 1), None, "since September 2025"),
        ("what did I change since december", date(2025, 12, 1), None, "since December 2025"),  # last year
    ],
)
def test_parse_window(text, since, commits, label):
    w = parse_window(text, TODAY)
    assert w is not None and (w.since, w.commits, w.label) == (since, commits, label)


@pytest.mark.parametrize(
    "text",
    ["what changed recently?", "when did I add retries?", "it may be in the decoder", "how does auth work"],
)
def test_no_window(text):
    assert parse_window(text, TODAY) is None


def test_default_window_is_two_weeks():
    w = default_window(TODAY)
    assert w.since == date(2026, 10, 5) and w.git_since == "2026-10-05" and w.label == "the last 2 weeks"
