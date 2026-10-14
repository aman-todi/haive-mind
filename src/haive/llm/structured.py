"""Structured output as an LCEL chain, per model:

    native   = model.with_structured_output(schema, method="function_calling")
    json     = messages + format instructions | model | extract_json | PydanticOutputParser
    repair   = the same, plus the parse error  (via .with_fallbacks(exception_key="error"))
    per_model = native.with_fallbacks([json.with_fallbacks([repair])])

and the models are joined with ``.with_fallbacks()`` down the role's chain. Models without tool
calling skip the native step.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ValidationError

from haive.llm.budget import BudgetExceeded, Meter
from haive.llm.models import LLM, LLMError

T = TypeVar("T", bound=BaseModel)

# Failures that a JSON-mode retry on the same model can fix.
PARSE_ERRORS: tuple[type[BaseException], ...] = (OutputParserException, ValidationError, ValueError)
# Failures of the native path that should fall back to JSON mode on the same model
# (e.g. a 400 because the provider doesn't support tool calling).
NATIVE_ERRORS: tuple[type[BaseException], ...] = (*PARSE_ERRORS, NotImplementedError, openai.BadRequestError)


class StructuredError(LLMError):
    pass


# ---------------------------------------------------------------- cheap local cleanup before parsing


def _largest_json_block(text: str) -> str | None:
    """Largest balanced {...} or [...] block, aware of strings."""
    best: tuple[int, int] | None = None
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, i, in_str, esc = 0, start, False, False
            end = None
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
                        end = i + 1
                        break
                i += 1
            if end is None:  # unbalanced: let the parser's partial-JSON handling close it
                end = len(text)
            if best is None or end - start > best[1] - best[0]:
                best = (start, end)
            start = text.find(opener, end)
    return text[best[0] : best[1]] if best else None


def extract_json(message: BaseMessage | str) -> str:
    """Strip prose and fences around a JSON object, and fix trailing commas / Python literals."""
    text = message.content if isinstance(message, BaseMessage) else message
    text = text if isinstance(text, str) else json.dumps(text)
    m = re.search(r"```(?:json|JSON)?\s*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    block = _largest_json_block(text) or text
    block = re.sub(r",\s*([}\]])", r"\1", block)
    block = re.sub(r"(?<![\w\"])True(?![\w\"])", "true", block)
    block = re.sub(r"(?<![\w\"])False(?![\w\"])", "false", block)
    return re.sub(r"(?<![\w\"])None(?![\w\"])", "null", block)


def unwrap(schema: type[T]) -> RunnableLambda:
    """Accept {"Answer": {...}} or a bare list for single-field schemas before validation."""

    def fix(text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(data, dict) and len(data) == 1 and schema.__name__ in data:
            data = data[schema.__name__]
        elif isinstance(data, list) and len(schema.model_fields) == 1:
            data = {next(iter(schema.model_fields)): data}
        return json.dumps(data)

    return RunnableLambda(fix)


def _require(schema: type[T]) -> RunnableLambda:
    def check(value: Any) -> T:
        if isinstance(value, schema):
            return value
        if isinstance(value, dict):
            return schema.model_validate(value)
        raise OutputParserException(f"model did not return a {schema.__name__}")

    return RunnableLambda(check)


# ---------------------------------------------------------------- chain construction


def build_chain(model: BaseChatModel, schema: type[T]) -> Runnable:
    """The structured-output chain for one model. Input: {"messages": [...]}."""
    parser = PydanticOutputParser(pydantic_object=schema)
    fmt = f"Schema: {schema.__name__}\n{parser.get_format_instructions()}\nRespond with the JSON object only."
    json_prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder("messages"), ("system", "{format}")]
    ).partial(format=fmt)
    repair_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("messages"),
            ("system", "{format}"),
            ("human", "That output was invalid: {error}\nReturn only the corrected JSON object."),
        ]
    ).partial(format=fmt)
    tail = RunnableLambda(extract_json) | unwrap(schema) | parser
    json_chain = (json_prompt | model | tail).with_fallbacks(
        [repair_prompt | model | tail], exceptions_to_handle=PARSE_ERRORS, exception_key="error"
    )
    try:
        try:
            native_model = model.with_structured_output(schema, method="function_calling")
        except (TypeError, ValueError):  # models whose signature has no `method`
            native_model = model.with_structured_output(schema)
    except NotImplementedError:  # no tool calling at all
        return json_chain
    native = RunnableLambda(lambda d: d["messages"]) | native_model | _require(schema)
    return native.with_fallbacks([json_chain], exceptions_to_handle=NATIVE_ERRORS)


def structured(llm: LLM, role: str, schema: type[T], messages: Any, meter: Meter) -> T:
    if isinstance(messages, str):
        messages = [HumanMessage(messages)]
    chain = llm.with_fallbacks(role, lambda m: build_chain(m, schema), key=schema)
    try:
        return chain.invoke({"messages": list(messages)}, config=llm.run_config(role, meter))
    except BudgetExceeded:
        raise
    except Exception as e:  # any provider failure degrades the node, never crashes the graph
        raise StructuredError(
            f"structured output failed for {schema.__name__}: {type(e).__name__}: {e}"
        ) from e
