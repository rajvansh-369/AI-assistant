"""The tool table must not drift.

`golden_declarations.json` is the exact TOOL_DECLARATIONS literal that lived in
main.py before the registry existed, extracted from the pre-refactor file rather
than retyped.  These tests are what makes it safe to have deleted the old
if/elif dispatch chain: if porting a tool changed a name, a required field or a
parameter type, this fails.

The golden file is a snapshot, not a spec.  Deliberately changing a schema means
updating it in the same commit — that is the point, it forces the change to be
visible in review.

    python -m pytest tests/ -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import actions.tools  # noqa: F401  — registers the tool table
from core import registry

GOLDEN_PATH = Path(__file__).parent / "golden_declarations.json"
GOLDEN: list[dict] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
GOLDEN_BY_NAME: dict[str, dict] = {d["name"]: d for d in GOLDEN}

VALID_TYPES = {"OBJECT", "STRING", "INTEGER", "NUMBER", "BOOLEAN", "ARRAY"}


def test_every_legacy_tool_is_registered():
    missing = sorted(set(GOLDEN_BY_NAME) - set(registry.names()))
    assert not missing, f"tools lost in the registry migration: {missing}"


def test_no_unexpected_tools():
    """A new tool is fine — it just has to be added to the golden file too."""
    extra = sorted(set(registry.names()) - set(GOLDEN_BY_NAME))
    assert not extra, (
        f"tools registered but absent from golden_declarations.json: {extra}. "
        "Add them to the snapshot if the addition is intentional."
    )


@pytest.mark.parametrize("name", sorted(GOLDEN_BY_NAME))
def test_declaration_matches_golden(name):
    tool = registry.get(name)
    assert tool is not None, f"{name} is not registered"
    assert tool.declaration() == GOLDEN_BY_NAME[name], (
        f"{name} schema drifted from the pre-registry declaration"
    )


@pytest.mark.parametrize("tool", registry.all_tools(), ids=lambda t: t.name)
def test_parameters_are_valid_schema(tool):
    params = tool.parameters
    assert params.get("type") == "OBJECT", f"{tool.name}: top level must be OBJECT"

    props = params.get("properties", {})
    assert isinstance(props, dict), f"{tool.name}: properties must be an object"

    for field, spec in props.items():
        assert spec.get("type") in VALID_TYPES, (
            f"{tool.name}.{field}: bad type {spec.get('type')!r}"
        )
        if spec["type"] == "ARRAY":
            assert "items" in spec, f"{tool.name}.{field}: ARRAY needs items"

    for field in params.get("required", []):
        assert field in props, (
            f"{tool.name}: required field {field!r} is not in properties"
        )


@pytest.mark.parametrize("tool", registry.all_tools(), ids=lambda t: t.name)
def test_every_tool_has_a_bounded_timeout(tool):
    """A tool with no timeout can hang the conversation forever."""
    assert tool.timeout is not None, f"{tool.name} has no timeout"
    assert 0 < tool.timeout <= 600, f"{tool.name}: implausible timeout {tool.timeout}"


@pytest.mark.parametrize("tool", registry.all_tools(), ids=lambda t: t.name)
def test_tool_takes_params_and_ctx(tool):
    import inspect

    sig = inspect.signature(tool.fn)
    assert len(sig.parameters) == 2, (
        f"{tool.name}: expected (params, ctx), got {list(sig.parameters)}"
    )
