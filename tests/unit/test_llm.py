import pytest
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel

from haive.config import Config
from haive.llm.budget import BudgetExceeded, Meter, daily_status
from haive.llm.models import LLM, LLMError, get_model, make_rate_limiter
from haive.llm.structured import StructuredError, extract_json, structured
from tests.fakes import FakeChatModel, api_error, factory_for


class Out(BaseModel):
    intent: str
    n: int = 0


def cfg_for() -> Config:
    cfg = Config()
    cfg.models.requests_per_minute = 0
    cfg.models.fast = ["m1", "m2"]
    return cfg


def make(models, db=None) -> LLM:
    return LLM(cfg_for(), db, factory_for(models))


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Sure! {"a": [1, 2,], "b": True, "c": None} hope that helps') == (
        '{"a": [1, 2], "b": true, "c": null}'
    )
    assert extract_json('prefix {"a": {"b": 1}') == '{"a": {"b": 1}'  # parser closes partial JSON


def test_json_mode_with_cleanup_records_usage(db):
    fake = FakeChatModel({"Out": ['text before {"intent": "lookup", "n": 2,} after']})
    m = Meter()
    assert structured(make(fake, db), "fast", Out, "q", m) == Out(intent="lookup", n=2)
    assert m.calls == 1 and fake.calls == ["Out"]
    assert db.calls_today() == 1
    row = db.usage_today()[0]
    assert (row["role"], row["model"], row["prompt_tokens"], row["completion_tokens"]) == (
        "fast",
        "fake-model",
        10,
        5,
    )
    assert m.log[0]["type"] == "llm" and m.log[0]["ok"]


def test_repair_feeds_error_back_then_falls_back():
    m1 = FakeChatModel({"Out": ["garbage", '{"n": "not-int"}']}, model_name="m1")
    m2 = FakeChatModel({"Out": [{"intent": "recall"}]}, model_name="m2")
    meter = Meter()
    assert structured(make({"m1": m1, "m2": m2}), "fast", Out, "q", meter).intent == "recall"
    assert m1.calls == ["Out", "Out"] and m2.calls == ["Out"]
    assert "That output was invalid" in m1.seen[1][-1].content
    assert meter.calls == 3 and [e["model"] for e in meter.log] == ["m1", "m1", "m2"]


def test_all_models_fail():
    with pytest.raises(StructuredError):
        structured(make(FakeChatModel({"Out": ["nope"]})), "fast", Out, "q", Meter())


@pytest.mark.parametrize("status", [429, 500, 502])
def test_api_errors_fall_back(status):
    m1 = FakeChatModel({"Out": [api_error(status)]}, model_name="m1")
    m2 = FakeChatModel({"Out": [{"intent": "fb"}]}, model_name="m2")
    meter = Meter()
    assert structured(make({"m1": m1, "m2": m2}), "fast", Out, "q", meter).intent == "fb"
    assert [e["ok"] for e in meter.log] == [False, True]


def test_native_tool_calling_used_when_supported():
    fake = FakeChatModel({"Out": [{"intent": "native", "n": 3}]}, supports_tools=True)
    m = Meter()
    assert structured(make(fake), "fast", Out, "q", m) == Out(intent="native", n=3)
    assert m.calls == 1


def test_native_failure_falls_back_to_json_mode_on_same_model():
    # the native call gets a 400 (e.g. the provider rejects tools); the JSON prompt path then succeeds
    fake = FakeChatModel({"Out": [api_error(400), '{"intent": "json"}']}, supports_tools=True)
    assert structured(make(fake), "fast", Out, "q", Meter()).intent == "json"


def test_budget_stops_before_sending():
    fake = FakeChatModel({"Out": ["garbage"]})
    meter = Meter(limit=1)
    with pytest.raises(BudgetExceeded):
        structured(make(fake), "fast", Out, "q", meter)
    assert meter.calls == 1 and len(fake.calls) == 1  # the second (repair) call was never sent


def test_text_and_fallback():
    m1 = FakeChatModel({"text": [api_error(503)]}, model_name="m1")
    m2 = FakeChatModel({"text": ["A short title"]}, model_name="m2")
    assert make({"m1": m1, "m2": m2}).text("fast", [HumanMessage("hi")], Meter()) == "A short title"
    with pytest.raises(LLMError):
        make(FakeChatModel({"text": [api_error(503)]})).text("fast", "hi", Meter())


def test_one_list_per_tier():
    cfg = cfg_for()
    llm = LLM(cfg, None, factory_for(FakeChatModel()))
    assert llm.model_ids() == ["m1", "m2"]  # tier "fast", free: every step uses this list
    cfg.models.tier = "strong"
    assert llm.model_ids()[0] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert llm.model_ids()[-1] == "openrouter/free"
    cfg.models.use_paid = True
    assert llm.model_ids() == ["deepseek/deepseek-v4-pro-0813", "z-ai/glm-5.2", "openrouter/auto"]
    cfg.models.tier = "fast"
    assert llm.model_ids()[0] == "deepseek/deepseek-v4.1-flash-1017"


def test_a_model_that_fails_a_call_is_skipped_for_the_rest_of_the_question():
    m1 = FakeChatModel({"text": [api_error(503), "late"]}, model_name="m1")
    m2 = FakeChatModel({"text": ["ok"]}, model_name="m2")
    llm = make({"m1": m1, "m2": m2})
    assert llm.text("router", "hi", Meter()) == "ok"
    assert llm.text("answer", "hi", Meter()) == "ok" and len(m1.calls) == 1  # same model for later steps
    llm.begin_question()
    assert llm.model_ids() == ["m1", "m2"]  # a new question tries the first model again


def test_get_model_wiring(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = Config()
    limiter = make_rate_limiter(20)
    model = get_model("nvidia/nemotron-3-super-120b-a12b:free", cfg, limiter)
    assert model.max_retries == 2  # the OpenAI client retries 429s honoring Retry-After
    assert model.rate_limiter is limiter and isinstance(limiter, InMemoryRateLimiter)
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.default_headers["X-Title"] == "haive-mind"
    assert make_rate_limiter(0) is None


def test_daily_status():
    assert daily_status(10, 1000) == "ok"
    assert daily_status(900, 1000) == "warn"
    assert daily_status(1000, 1000) == "over"


def test_meter_records_client_retries_inside_a_call():
    # The OpenAI client retries timeouts and 429s inside one call; --trace should say so.
    import logging

    fake = FakeChatModel({"text": ["hi"]})
    client_log = logging.getLogger("openai._base_client")

    def slow_reply(*a, **kw):
        client_log.debug("Encountered a timeout exception: %s", "ReadTimeout")
        client_log.debug("Encountered an HTTP status error: %i", 429)
        return orig(*a, **kw)

    orig, fake._generate = fake._generate, slow_reply
    meter = Meter()
    LLM(Config(), None, factory_for(fake)).text("fast", "hi", meter)
    assert meter.log[-1]["failed_attempts"] == ["timeout", "HTTP 429"]


def test_model_that_404s_is_skipped_afterwards():
    # A removed model (or one whose only provider is down) answers 404: don't pay for it on every call.
    m1 = FakeChatModel({"text": [api_error(404)]}, model_name="m1")
    m2 = FakeChatModel({"text": ["ok"]}, model_name="m2")
    llm = make({"m1": m1, "m2": m2})
    assert llm.text("fast", "hi", Meter()) == "ok" and len(m1.calls) == 1
    assert llm.text("fast", "hi", Meter()) == "ok" and len(m1.calls) == 1  # not tried again
    llm.begin_question()
    assert llm.model_ids() == ["m2"]  # a 404 lasts beyond the question
    m3 = FakeChatModel({"text": [api_error(503)]}, model_name="m3")
    busy = make({"m1": m3, "m2": m2})
    busy.text("fast", "hi", Meter())
    busy.begin_question()
    assert busy.model_ids() == ["m1", "m2"]  # a 503 is transient: next question tries it again
