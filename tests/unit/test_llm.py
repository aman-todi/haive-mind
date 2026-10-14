import pytest
from pydantic import BaseModel

from haive.config import Config
from haive.llm.budget import BudgetExceeded, Meter, RateLimiter, daily_status
from haive.llm.models import LLM
from haive.llm.structured import StructuredError, repair_json, structured
from tests.fakes import FakeChatModel, FakeHTTPError, factory_for


class Out(BaseModel):
    intent: str
    n: int = 0


def make(script_or_models, db=None, sleeps=None):
    cfg = Config()
    cfg.models.requests_per_minute = 0
    cfg.models.fast, cfg.models.fallback = "m1", ["m2"]
    sleeps = sleeps if sleeps is not None else []
    models = script_or_models if isinstance(script_or_models, dict) and "m1" in script_or_models else None
    fake = None if models else FakeChatModel(script_or_models)
    return LLM(cfg, db, factory_for(models or fake), sleep=sleeps.append), fake


def test_repair_json_variants():
    assert repair_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert repair_json('Sure! Here it is: {"a": [1, 2,], "b": True, "c": None} hope that helps') == {
        "a": [1, 2],
        "b": True,
        "c": None,
    }
    assert repair_json('{"a": {"b": 1}') == {"a": {"b": 1}}
    assert repair_json("{'a': 'x'}") == {"a": "x"}
    with pytest.raises(ValueError):
        repair_json("no json here")


def test_structured_json_mode_and_repair(db):
    llm, fake = make({"Out": ['text before {"intent": "lookup", "n": 2,} after']}, db)
    m = Meter()
    assert structured(llm, "fast", Out, "q", m) == Out(intent="lookup", n=2)
    assert m.calls == 1 and fake.calls == ["Out"]
    assert db.calls_today() == 1


def test_structured_retries_with_error_then_falls_back():
    m1 = FakeChatModel({"Out": ["garbage", '{"n": "not-int"}']})
    m2 = FakeChatModel({"Out": [{"intent": "recall"}]})
    llm, _ = make({"m1": m1, "m2": m2})
    meter = Meter()
    assert structured(llm, "fast", Out, "q", meter).intent == "recall"
    assert m1.calls == ["Out", "Out"] and m2.calls == ["Out"]
    assert "invalid" in m1.messages[1][-1].content
    assert meter.calls == 3


def test_structured_all_fail():
    llm, _ = make({"Out": ["nope"]})
    with pytest.raises(StructuredError):
        structured(llm, "fast", Out, "q", Meter())


def test_429_backoff_respects_retry_after_and_caps():
    sleeps = []
    llm, fake = make(
        {"Out": [FakeHTTPError(429, "3"), FakeHTTPError(429, "100"), {"intent": "x"}]}, sleeps=sleeps
    )
    assert structured(llm, "fast", Out, "q", Meter()).intent == "x"
    assert sleeps == [3.0, 20.0]


def test_429_exhausted_moves_to_fallback():
    m1 = FakeChatModel({"Out": [FakeHTTPError(429)]})
    m2 = FakeChatModel({"Out": [{"intent": "fb"}]})
    sleeps = []
    llm, _ = make({"m1": m1, "m2": m2}, sleeps=sleeps)
    assert structured(llm, "fast", Out, "q", Meter()).intent == "fb"
    assert sleeps == [2.0, 4.0]


def test_5xx_falls_back():
    m1 = FakeChatModel({"Out": [FakeHTTPError(502)]})
    m2 = FakeChatModel({"Out": [{"intent": "fb"}]})
    llm, _ = make({"m1": m1, "m2": m2})
    assert structured(llm, "fast", Out, "q", Meter()).intent == "fb"


def test_meter_budget():
    llm, _ = make({"Out": ["garbage"]})
    meter = Meter(limit=1)
    with pytest.raises(BudgetExceeded):
        structured(llm, "fast", Out, "q", meter)
    assert meter.calls == 1


def test_native_structured_output_used_when_available():
    class Native(FakeChatModel):
        def with_structured_output(self, schema, **kw):
            outer = self

            class R:
                def invoke(self, msgs):
                    outer.calls.append("native")
                    return {"parsed": schema(intent="native"), "raw": None}

            return R()

    llm, _ = make({"m1": Native(), "m2": Native()})
    assert structured(llm, "fast", Out, "q", Meter()).intent == "native"


def test_rate_limiter_window():
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    rl = RateLimiter(2, clock=lambda: t[0], sleep=sleep)
    rl.acquire()
    rl.acquire()
    rl.acquire()
    assert len(slept) == 1 and 59 < slept[0] <= 60.1


def test_daily_status():
    assert daily_status(10, 1000) == "ok"
    assert daily_status(900, 1000) == "warn"
    assert daily_status(1000, 1000) == "over"
