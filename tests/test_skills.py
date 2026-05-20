"""Tests for the low-coupling skill module."""

import pytest

from langdeep import (
    Skill,
    SkillAdapters,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillRegistry,
    load_skill_manifest,
    load_skill_manifest_file,
    skill,
    skill_registry,
)
from langdeep.core.errors import ConfigurationError, SkillNotFoundError
from langdeep.core.registry.tool_registry import tool_registry


def setup_function():
    from tests.conftest import clean_registries

    clean_registries()


class ExampleSkill(Skill):
    def __init__(self):
        super().__init__(
            SkillManifest(
                name="example",
                version="1.0.0",
                description="Example skill",
                owner="tests",
                tags=["demo"],
                required_permissions=["tools.invoke"],
                capabilities=[
                    SkillCapability(kind="tool", name="lookup"),
                    {"kind": "mcp", "name": "lookup_tool"},
                    {"kind": "a2a", "name": "lookup_skill"},
                ],
            )
        )
        self.validated = False
        self.deactivated = False

    def validate(self, context=None):
        self.validated = True
        return ["optional dependency not configured"]

    def activate(self, context=None):
        super().activate(context)
        return SkillAdapters(
            tools={"lookup": object()},
            mcp={"tools": ["lookup_tool"]},
            a2a={"skills": ["lookup_skill"]},
        )

    def deactivate(self, context=None):
        self.deactivated = True
        super().deactivate(context)


def test_skill_manifest_validates_and_round_trips():
    manifest = SkillManifest.from_dict(
        {
            "schema_version": "langdeep.skill.v1",
            "name": "enterprise_search",
            "version": "0.1.0",
            "tags": ["search", "internal"],
            "required_permissions": ["network.egress"],
            "capabilities": [
                {
                    "kind": "tool",
                    "name": "search",
                    "input_schema": {"type": "object"},
                }
            ],
            "extensions": {"x-company": {"tier": "internal"}},
        }
    )

    assert manifest.name == "enterprise_search"
    assert manifest.capabilities[0].kind == "tool"
    assert manifest.as_dict()["extensions"]["x-company"]["tier"] == "internal"


def test_skill_manifest_rejects_unknown_schema_and_capability_kind():
    with pytest.raises(ConfigurationError):
        SkillManifest(name="bad", version="1", schema_version="unknown")

    with pytest.raises(ConfigurationError):
        SkillCapability(kind="unknown", name="bad")


def test_skill_registry_is_low_coupling_until_activation():
    audit_events = []
    registry = SkillRegistry.for_namespace("tenant-a")
    registry.register("example", ExampleSkill)

    assert registry.list_skills() == ["example"]
    assert tool_registry.list_tools() == []

    manifest = registry.get_manifest("example")
    assert manifest.name == "example"
    assert tool_registry.list_tools() == []

    adapters = registry.activate(
        "example",
        SkillContext(
            tenant_id="acme",
            namespace="tenant-a",
            user_id="user-1",
            audit_sink=audit_events,
        ),
    )

    assert "lookup" in adapters.tools
    assert registry.get_active_adapters("example") is adapters
    assert registry.get_skill("example").validated is True
    assert registry.get_skill("example").active is True
    assert tool_registry.list_tools() == []
    assert [event["event_type"] for event in audit_events] == [
        "skill.validate",
        "skill.activate",
    ]
    assert audit_events[0]["tenant_id"] == "acme"

    registry.deactivate("example", SkillContext(namespace="tenant-a", audit_sink=audit_events))
    assert registry.get_active_adapters("example") is None
    assert registry.get_skill("example").deactivated is True
    assert registry.get_skill("example").active is False
    assert audit_events[-1]["event_type"] == "skill.deactivate"


def test_skill_registry_namespaces_and_duplicate_rejection():
    first = SkillRegistry.for_namespace("first")
    second = SkillRegistry.for_namespace("second")

    first.register("example", ExampleSkill)
    assert first.list_skills() == ["example"]
    assert second.list_skills() == []

    with pytest.raises(ConfigurationError):
        first.register("example", ExampleSkill, replace=False)

    with pytest.raises(SkillNotFoundError):
        second.get_skill("example")


def test_skill_decorator_registers_factory_without_instantiating():
    created = []

    @skill(
        name="decorated",
        version="0.1.0",
        capabilities=[{"kind": "workflow", "name": "triage"}],
    )
    def build_skill():
        created.append(True)
        return Skill(
            SkillManifest(
                name="decorated",
                version="0.1.0",
                capabilities=[{"kind": "workflow", "name": "triage"}],
            )
        )

    assert skill_registry.list_skills() == ["decorated"]
    assert created == []
    assert skill_registry.get_manifest("decorated").capabilities[0].name == "triage"
    assert created == []

    instance = skill_registry.get_skill("decorated")
    assert instance.manifest.name == "decorated"
    assert created == [True]


def test_load_skill_manifest_from_mapping_and_file(tmp_path):
    data = {
        "name": "loaded",
        "version": "0.1.0",
        "capabilities": [{"kind": "prompt", "name": "support"}],
    }

    manifest = load_skill_manifest(data)
    assert manifest.name == "loaded"

    path = tmp_path / "skill.yml"
    path.write_text(
        "\n".join(
            [
                "name: file_loaded",
                "version: 0.1.0",
                "capabilities:",
                "  - kind: resource",
                "    name: handbook",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_skill_manifest_file(path)
    assert loaded.name == "file_loaded"
    assert loaded.capabilities[0].kind == "resource"
