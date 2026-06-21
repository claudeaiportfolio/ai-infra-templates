"""Pure-logic tests for agent-core (no Anthropic/MCP/network calls)."""

import pytest

from agent_core import Auth0M2MClient, AuthError, SkillLoader, ToolSchema
from agent_core.auth import _CachedToken


def test_tool_schema_to_anthropic():
    schema = ToolSchema(
        name="retrieve",
        description="search the corpus",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    assert schema.to_anthropic() == {
        "name": "retrieve",
        "description": "search the corpus",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }


def test_skill_loader_composes_inventory_and_body(tmp_path):
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: a demo skill\n---\nUse this when demoing.\n"
    )
    loader = SkillLoader(skills_dir=tmp_path / "skills")
    composed = loader.compose_system_prompt("BASE PROMPT", "any question")
    assert "BASE PROMPT" in composed
    assert "demo" in composed  # inventory line
    assert "Use this when demoing." in composed  # body (always_load default)


def test_skill_loader_no_skills_returns_base(tmp_path):
    loader = SkillLoader(skills_dir=tmp_path / "does-not-exist")
    assert loader.compose_system_prompt("BASE", "q") == "BASE"


def test_auth_from_env_missing_raises(monkeypatch):
    for var in (
        "AUTH0_DOMAIN",
        "AUTH0_M2M_CLIENT_ID",
        "AUTH0_M2M_CLIENT_SECRET",
        "AUTH0_M2M_AUDIENCE",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AuthError, match="Missing required Auth0 M2M"):
        Auth0M2MClient.from_env()


def test_cached_token_freshness():
    import time

    fresh = _CachedToken(access_token="t", expires_at=time.monotonic() + 3600)
    stale = _CachedToken(access_token="t", expires_at=time.monotonic() + 10)  # < 60s buffer
    assert fresh.is_fresh() is True
    assert stale.is_fresh() is False
