"""Structured output: native (tool calling / json_schema) -> JSON-mode prompt -> JSON repair.

Each model gets two tries (the second with the validation error fed back); then we move down the
fallback chain. Models whose native structured output fails are remembered and skipped next time.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from haive.llm.budget import BudgetExceeded, Meter
from haive.llm.models import LLM, LLMError, is_rate_limit, is_server_error

T = TypeVar("T", bound=BaseModel)


class StructuredError(LLMError):
    pass


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json|JSON)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def _largest_json_block(text: str) -> str | None:
    """Largest balanced {...} or [...] block, aware of strings."""
    best: tuple[int, int] | None = None
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, i, in_str, esc = 0, start, False, False
            while i < len(text):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        if best is None or i + 1 - start > best[1] - best[0]:
                            best = (start, i + 1)
                        break
                i += 1
            else:
                # unbalanced: take the rest and try closing it later
                if best is None:
                    best = (start, len(text))
            start = text.find(opener, i + 1) if i < len(text) else -1
    return text[best[0] : best[1]] if best else None


def _fix_common(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    s = re.sub(r"(?<![\w\"])True(?![\w\"])", "true", s)
    s = re.sub(r"(?<![\w\"])False(?![\w\"])", "false", s)
    s = re.sub(r"(?<![\w\"])None(?![\w\"])", "null", s)
    s = re.sub(r"//[^\n\"]*\n", "\n", s)  # line comments outside strings (best effort)
    # close unbalanced brackets
    opens = s.count("{") - s.count("}")
    sq = s.count("[") - s.count("]")
    if sq > 0:
        s += "]" * sq
    if opens > 0:
        s += "}" * opens
    return s


def repair_json(text: str) -> Any:
    """Parse JSON out of model text. Raises ValueError if nothing usable is found."""
    text = _strip_fences(text.strip())
    for candidate in (text, _largest_json_block(text)):
        if not candidate:
            continue
        for fn in (lambda x: x, _fix_common):
            try:
                return json.loads(fn(candidate))
            except json.JSONDecodeError:
                continue
    # last resort: single-quoted JSON
    block = _largest_json_block(text)
    if block:
        try:
            return json.loads(_fix_common(re.sub(r"'", '"', block)))
        except json.JSONDecodeError:
            pass
    raise ValueError("no JSON object found in model output")


def parse_as(schema: type[T], text: str) -> T:
    data = repair_json(text)
    if isinstance(data, list) and len(schema.model_fields) == 1:
        data = {next(iter(schema.model_fields)): data}
    if isinstance(data, dict) and len(data) == 1 and schema.__name__ in data:
        data = data[schema.__name__]  # {"Answer": {...}}
    return schema.model_validate(data)


def json_instructions(schema: type[BaseModel]) -> str:
    return (
        f"Schema: {schema.__name__}\n"
        "Respond with ONLY one JSON object that validates against this JSON schema. "
        "No prose, no markdown fences.\n" + json.dumps(schema.model_json_schema(), separators=(",", ":"))
    )


def _as_messages(messages: Any) -> list:
    from langchain_core.messages import HumanMessage

    if isinstance(messages, str):
        return [HumanMessage(messages)]
    return list(messages)


def structured(llm: LLM, role: str, schema: type[T], messages: Any, meter: Meter) -> T:
    from langchain_core.messages import HumanMessage, SystemMessage

    msgs = _as_messages(messages)
    errors: list[str] = []
    for model_id in llm.chain(role):
        model = llm.model(model_id)
        # 1) native structured output
        if model_id not in llm.no_native and hasattr(model, "with_structured_output"):
            try:
                runnable = model.with_structured_output(schema, method="function_calling", include_raw=True)
                out = llm.invoke_once(role, model_id, runnable, msgs, meter)
                parsed = out.get("parsed") if isinstance(out, dict) else out
                if isinstance(parsed, schema):
                    return parsed
                if isinstance(parsed, dict):
                    return schema.model_validate(parsed)
                raw = out.get("raw") if isinstance(out, dict) else None
                if raw is not None and getattr(raw, "content", None):
                    try:
                        return parse_as(schema, str(raw.content))
                    except (ValueError, ValidationError):
                        pass
                llm.no_native.add(model_id)
            except BudgetExceeded:
                raise
            except NotImplementedError:
                llm.no_native.add(model_id)
            except Exception as e:
                if is_rate_limit(e) or is_server_error(e):
                    errors.append(f"{model_id}: {e}")
                    continue  # this model is struggling; fall back
                llm.no_native.add(model_id)
                errors.append(f"{model_id} native: {e}")
        # 2) JSON-mode prompt + repair, twice (second time with the error)
        jmsgs = [*msgs, SystemMessage(json_instructions(schema))]
        for attempt in range(2):
            try:
                msg = llm.invoke_once(role, model_id, model, jmsgs, meter)
            except BudgetExceeded:
                raise
            except Exception as e:
                errors.append(f"{model_id}: {e}")
                break
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            try:
                return parse_as(schema, content)
            except (ValueError, ValidationError) as e:
                err = str(e)[:800]
                errors.append(f"{model_id} invalid output: {err[:200]}")
                if attempt == 0:
                    jmsgs = [
                        *jmsgs,
                        HumanMessage(
                            f"That output was invalid: {err}\nReturn only the corrected JSON object."
                        ),
                    ]
    raise StructuredError(f"structured output failed for {schema.__name__}: " + " | ".join(errors[-4:]))
