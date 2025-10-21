"""Utility to convert Python callables into OpenAI tool definitions."""

from __future__ import annotations

import dataclasses
import enum
import inspect
import math
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, time
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from openai.types.chat import (
    ChatCompletionFunctionToolParam,
)


def callables_to_tool_schemas(
    callables: Iterable[Callable[..., Any]],
) -> list[ChatCompletionFunctionToolParam]:
    """Translate Python callables into OpenAI tool function dictionaries.

    Returns a list you can pass directly to the `tools=...` Chat Completions API parameter.
    """
    # tools: list[dict[str, Any]] = []
    tools: list[ChatCompletionFunctionToolParam] = []

    for func in callables:
        if not callable(func):
            msg = f"Expected callable, received {type(func)!r}"
            raise TypeError(msg)

        signature = inspect.signature(func)
        try:
            type_hints = get_type_hints(func, include_extras=True)
        except Exception:
            type_hints = {}

        # Function-level description and name
        func_doc = inspect.getdoc(func) or "No description provided."
        name = _normalize_name(getattr(func, "__name__", "tool"))

        parameters_schema = _build_parameters_schema(signature, type_hints, func_doc)

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": func_doc,
                    "parameters": parameters_schema,
                },
            }
        )

    return tools


# -----------------
# Schema construction
# -----------------


def _build_parameters_schema(
    signature: inspect.Signature,
    type_hints: dict[str, Any],
    func_doc: str,
) -> dict[str, Any]:
    """Build a strict JSON Schema for the function parameters."""
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for param_name, param in signature.parameters.items():
        # Skip *args and **kwargs: these are not representable in tool schemas.
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = type_hints.get(param_name, Any)
        schema = _type_to_schema(annotation)

        # Use the *function docstring* for each parameter description.
        schema["description"] = func_doc

        # Defaults: include only if JSON-compatible (and finite if float).
        if param.default is not inspect._empty:
            default_value = _coerce_json_value(param.default)
            if _is_json_compatible(default_value):
                # If default is None, ensure the schema allows null.
                if default_value is None and not _schema_allows_null(schema):
                    # Build the wrapped schema and re-apply the description in one step
                    schema = {
                        "anyOf": [schema, {"type": "null"}],
                        "description": func_doc,
                    }
                schema.setdefault("default", default_value)
        else:
            required.append(param_name)

        properties[param_name] = schema

    parameter_block: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        # Strict JSON Schema: disallow extra keys not defined in properties
        "additionalProperties": False,
    }
    if required:
        parameter_block["required"] = required
    return parameter_block


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    """Translate a Python type annotation into a JSON Schema fragment."""
    base_annotation, metadata = _unwrap_annotated(annotation)
    annotation = base_annotation
    origin = get_origin(annotation)
    args = get_args(annotation)
    NoneType = type(None)

    # Literal[...] -> enum
    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        # If all enum values share a JSON type, set it (safer for strict validation).
        if all(isinstance(v, bool) for v in values):
            schema["type"] = "boolean"
        elif all(isinstance(v, str) for v in values):
            schema["type"] = "string"
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            schema["type"] = "integer"
        elif all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        ):
            schema["type"] = "number"
        result = schema

    # Any or empty -> default to string (conservative, strict)
    elif annotation in (inspect._empty, Any):
        result = {"type": "string"}

    else:
        primitive_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        if annotation in primitive_map:
            result = {"type": primitive_map[annotation]}

        elif annotation is bytes:
            result = {"type": "string", "contentEncoding": "base64"}

        elif annotation is datetime:
            result = {"type": "string", "format": "date-time"}

        elif annotation is date:
            result = {"type": "string", "format": "date"}

        elif annotation is time:
            result = {"type": "string", "format": "time"}

        elif inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            values = [member.value for member in annotation]
            schema = {"enum": values}
            value_types = {type(v) for v in values}
            # Only set "type" if homogeneous; otherwise keep just enum.
            if value_types == {str}:
                schema["type"] = "string"
            elif value_types == {int}:
                schema["type"] = "integer"
            elif value_types <= {int, float}:
                schema["type"] = "number"
            elif value_types == {bool}:
                schema["type"] = "boolean"
            result = schema

        elif _is_typed_dict(annotation):
            # Resolve forward refs & Annotated extras
            try:
                field_hints = get_type_hints(annotation, include_extras=True)
            except Exception:
                field_hints = getattr(annotation, "__annotations__", {})
            required_keys = set(getattr(annotation, "__required_keys__", set()))
            td_properties: dict[str, Any] = {}
            td_required_fields: list[str] = []
            for key, value in field_hints.items():
                td_properties[key] = _type_to_schema(value)
                if key in required_keys:
                    td_required_fields.append(key)
            schema = {
                "type": "object",
                "properties": td_properties,
                "additionalProperties": False,
            }
            if td_required_fields:
                schema["required"] = td_required_fields
            result = schema

        elif dataclasses.is_dataclass(annotation):
            try:
                field_hints = get_type_hints(annotation, include_extras=True)
            except Exception:
                field_hints = {}
            dc_properties: dict[str, Any] = {}
            dc_required_fields: list[str] = []
            for field in dataclasses.fields(annotation):
                field_type = field_hints.get(field.name, Any)
                dc_properties[field.name] = _type_to_schema(field_type)
                if (
                    field.default is dataclasses.MISSING
                    and getattr(field, "default_factory", dataclasses.MISSING)
                    is dataclasses.MISSING
                ):
                    dc_required_fields.append(field.name)
            schema = {
                "type": "object",
                "properties": dc_properties,
                "additionalProperties": False,
            }
            if dc_required_fields:
                schema["required"] = dc_required_fields
            result = schema

        elif origin in {list, set, frozenset, Sequence}:
            item_type = args[0] if args else Any
            schema = {"type": "array", "items": _type_to_schema(item_type)}
            if origin in {set, frozenset}:
                schema["uniqueItems"] = True
            result = schema

        elif origin is tuple:
            # tuple[], tuple[T, ...], tuple[T1, T2, ...]
            if not args:
                result = {"type": "array", "items": {"type": "string"}}
            elif len(args) == 2 and args[1] is Ellipsis:
                result = {"type": "array", "items": _type_to_schema(args[0])}
            else:
                prefix_items = [_type_to_schema(arg) for arg in args]
                result = {
                    "type": "array",
                    "prefixItems": prefix_items,
                    "minItems": len(prefix_items),
                    "maxItems": len(prefix_items),
                }

        elif origin in {dict, Mapping}:
            # Mapping[str, V] or Mapping[Any, V] -> dynamic object with value schema
            value_type = args[1] if len(args) >= 2 else Any
            result = {
                "type": "object",
                "additionalProperties": _type_to_schema(value_type),
            }

        elif origin in {Union, types.UnionType}:
            has_none = any(arg is NoneType for arg in args)
            non_none = [arg for arg in args if arg is not NoneType]
            if not non_none:
                # Degenerate case: Union[None]
                base: dict[str, Any] = {"type": "null"}
            elif len(non_none) == 1:
                base = _type_to_schema(non_none[0])
            else:
                # Collapse pure numeric unions to number
                if set(non_none) <= {int, float}:
                    base = {"type": "number"}
                else:
                    base = {"anyOf": [_type_to_schema(arg) for arg in non_none]}
            if has_none:
                # Safely append to anyOf if present and is a list; otherwise wrap.
                anyof_val = base.get("anyOf")
                if isinstance(anyof_val, list):
                    anyof_val.append({"type": "null"})
                else:
                    base = {"anyOf": [base, {"type": "null"}]}
            result = base

        elif isinstance(annotation, type):
            # Bare containers (non-parameterized) -> sensible defaults
            from collections.abc import Mapping as _MappingABC
            from collections.abc import Sequence as _SequenceABC

            if annotation in {list, tuple} or issubclass(annotation, _SequenceABC):
                result = {"type": "array", "items": {"type": "string"}}
            elif annotation in {set, frozenset}:
                result = {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                }
            elif annotation is dict or issubclass(annotation, _MappingABC):
                result = {"type": "object", "additionalProperties": {"type": "string"}}
            else:
                result = {"type": "string"}

        else:
            result = {"type": "string"}

    # Carry over any Annotated[...] metadata by merging as overrides.
    if metadata:
        result = _apply_annotated_metadata(result, metadata)
    return result


# -----------------
# Utilities
# -----------------


def _apply_annotated_metadata(
    schema: dict[str, Any], metadata: Sequence[Any]
) -> dict[str, Any]:
    """Merge metadata (strings -> description; mappings/dataclasses/objects -> deep-merge)."""
    updated: dict[str, Any] = dict(schema)
    for meta in metadata:
        if isinstance(meta, str):
            updated["description"] = meta
            continue
        if dataclasses.is_dataclass(meta) and not isinstance(meta, type):
            # Only convert dataclass *instances* to dicts
            meta_value = dataclasses.asdict(meta)
        elif isinstance(meta, Mapping):
            meta_value = dict(meta)
        elif hasattr(meta, "__dict__") and not isinstance(meta, type):
            meta_value = {k: v for k, v in vars(meta).items() if not k.startswith("_")}
        else:
            # Unknown metadata type; skip
            continue
        updated = _merge_dicts(updated, cast(Mapping[str, Any], meta_value))
    return updated


def _merge_dicts(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(cast(dict[str, Any], merged[key]), value)
        else:
            merged[key] = value
    return merged


def _is_typed_dict(annotation: Any) -> bool:
    return (
        isinstance(annotation, type)
        and hasattr(annotation, "__required_keys__")
        and hasattr(annotation, "__optional_keys__")
    )


def _normalize_name(name: str) -> str:
    filtered = [c if c.isalnum() or c in {"_", "-"} else "_" for c in name]
    cleaned = "".join(filtered) or "tool"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned[:64]


def _unwrap_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Unwrap typing.Annotated[T, ...] into (T, metadata...)."""
    metadata: list[Any] = []
    current = annotation
    while get_origin(current) is Annotated:
        args = get_args(current)
        if not args:
            break
        current = args[0]
        metadata.extend(args[1:])
    return current, tuple(metadata)


def _schema_allows_null(schema: dict[str, Any]) -> bool:
    """Return True if the schema already allows nulls."""
    if schema.get("type") == "null":
        return True
    # Some schemas may use anyOf (we don't generate oneOf here intentionally).
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        return any(
            isinstance(s, dict) and s.get("type") == "null" for s in schema["anyOf"]
        )
    return False


def _is_json_compatible(value: Any) -> bool:
    """Conservative JSON-compatibility check (rejects NaN/Infinity)."""
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_is_json_compatible(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_compatible(item)
            for key, item in value.items()
        )
    # bytes and other objects are not used as defaults in schema
    return False


def _coerce_json_value(value: Any) -> Any:
    """Coerce Python containers into JSON-compatible forms (e.g., tuples → lists)."""
    if isinstance(value, tuple):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, list):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _coerce_json_value(item) for key, item in value.items()}
    return value
