"""Skill loading — discovers SKILL.md files and exposes them to the agent.

Follows the progressive-disclosure pattern from Anthropic's skills spec:

  1. Metadata (name + description from YAML frontmatter) is always loaded.
  2. Body (everything below the frontmatter) is loaded into context when
     the skill is selected for the task at hand.
  3. Bundled resources (scripts/, references/, assets/ alongside SKILL.md)
     are not loaded by this module — the body itself can reference them
     by relative path, and the agent loads them on demand if needed.

With the default `always_load` selector, every discovered skill loads. The
architecture supports conditional loading via a custom selector predicate —
that's just a different predicate, not a rewrite.

Why we built it this way rather than inlining SKILL.md into the system
prompt:

  - Eval comparison: with-skill vs without-skill is a single boolean
    toggle, not two diverged prompt files that have to be kept in sync.
  - Future-proof: when the agent grows a second skill (e.g. a separate
    skill for audit-only investigations), no rewrite needed — drop a new
    SKILL.md into the skills/ directory.
  - Faithful to how Claude actually consumes skills: the YAML frontmatter
    is loaded separately from the body, and the body is treated as
    user-authored guidance, not as part of the agent's "core" identity.

Frontmatter format follows the skill-creator spec:

    ---
    name: skill-name
    description: When to trigger, what it does.
    ---

    # Body in Markdown ...
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Parses the YAML frontmatter block at the top of a SKILL.md.
# We don't pull in PyYAML for two reasons:
#   1. Skills only use string scalars — name and description — so the
#      grammar is tiny and a regex is faithful.
#   2. Avoids an extra dependency for what's a 10-line problem.
# If skills ever grow nested frontmatter (compatibility blocks, etc.)
# this is the place to swap in PyYAML.
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<frontmatter>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)
_FRONTMATTER_FIELD_RE = re.compile(
    r"^(?P<key>[A-Za-z_][A-Za-z0-9_-]*):\s*(?P<value>.*)$",
)


@dataclass(frozen=True)
class Skill:
    """A loaded SKILL.md, parsed into frontmatter + body."""

    name: str
    description: str
    body: str
    path: Path

    def preamble_line(self) -> str:
        """One-line summary for the always-loaded skills inventory."""
        return f"- **{self.name}**: {self.description}"


class SkillLoadError(RuntimeError):
    """A SKILL.md was malformed or missing required frontmatter fields."""


def parse_skill(path: Path) -> Skill:
    """Parse a single SKILL.md file.

    Required frontmatter fields: name, description.
    Anything else in frontmatter is ignored today; if we add optional
    fields later (e.g. compatibility constraints) this is the place.
    """
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise SkillLoadError(
            f"{path}: missing or malformed YAML frontmatter. "
            "Expected `---\\nname: ...\\ndescription: ...\\n---` at top of file."
        )

    fields: dict[str, str] = {}
    for line in match["frontmatter"].splitlines():
        line = line.strip()
        if not line:
            continue
        field_match = _FRONTMATTER_FIELD_RE.match(line)
        if not field_match:
            # Multi-line YAML values (continuation lines) — appended to the
            # most recently seen key. Skills use this for long descriptions.
            if fields:
                last_key = next(reversed(fields))
                fields[last_key] = (fields[last_key] + " " + line).strip()
            continue
        fields[field_match["key"]] = field_match["value"].strip()

    name = fields.get("name")
    description = fields.get("description")
    if not name:
        raise SkillLoadError(f"{path}: missing required `name` field in frontmatter")
    if not description:
        raise SkillLoadError(f"{path}: missing required `description` field in frontmatter")

    return Skill(
        name=name,
        description=description,
        body=match["body"].strip(),
        path=path,
    )


SkillSelector = Callable[[Skill, str], bool]
"""A predicate: given a loaded skill and the user's question, return
True if the skill's body should be loaded into context for this turn.

The trivial selector (`lambda s, q: True`) always loads every discovered
skill — appropriate when the agent has one or two always-relevant skills.

For larger skill libraries, swap in a classifier that reads the skill's
description and the question, then returns True only for matching skills.
That classifier is itself a small Claude call — not built today, but the
shape of the API is designed to accept it without changes."""


def always_load(_skill: Skill, _question: str) -> bool:
    """Default selector — every discovered skill loads on every turn."""
    return True


@dataclass
class SkillLoader:
    """Discovers SKILL.md files under a root directory and exposes them.

    Discovery is non-recursive in the sense that we look for skills at
    one level of nesting: `<skills_dir>/<skill-name>/SKILL.md`. This is
    the layout the skill-creator spec produces and matches how skills
    are distributed on Anthropic's platform.
    """

    skills_dir: Path
    selector: SkillSelector = field(default=always_load)
    _skills: list[Skill] = field(default_factory=list, init=False)
    _loaded: bool = field(default=False, init=False)

    def discover(self) -> list[Skill]:
        """Find and parse every SKILL.md under `skills_dir`.

        Idempotent — safe to call multiple times. Raises SkillLoadError
        if any discovered file is malformed; that's deliberate. Better to
        fail loudly at startup than to silently drop a misnamed skill.
        """
        if self._loaded:
            return self._skills

        if not self.skills_dir.exists():
            # Empty skills directory is fine — agent just runs without skills.
            self._loaded = True
            return []

        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            self._skills.append(parse_skill(skill_md))

        self._loaded = True
        return self._skills

    @property
    def skills(self) -> list[Skill]:
        if not self._loaded:
            self.discover()
        return self._skills

    def inventory_preamble(self) -> str:
        """Always-loaded text describing which skills exist.

        This is the frontmatter-level disclosure: the model always knows
        what skills are available and when to expect them to apply, even
        when the body isn't loaded.
        """
        if not self.skills:
            return ""
        lines = ["Skills available for this agent:"]
        lines.extend(s.preamble_line() for s in self.skills)
        return "\n".join(lines)

    def selected_bodies(self, question: str) -> list[Skill]:
        """Return the skills whose bodies should load into context for `question`.

        Today, with `always_load`, this returns every discovered skill.
        With a real selector, it returns only matching skills.
        """
        return [s for s in self.skills if self.selector(s, question)]

    def compose_system_prompt(self, base_prompt: str, question: str) -> str:
        """Build the full system prompt: base + skill inventory + selected bodies.

        Structure:

            <base prompt — role, tool inventory, basic guidance>

            <skill inventory — always present>

            <selected skill bodies — present when their selector matched>

        Each skill body is fenced with its name for clarity, so the
        model can see which guidance came from which skill.
        """
        parts = [base_prompt.rstrip()]

        inventory = self.inventory_preamble()
        if inventory:
            parts.append(inventory)

        for skill in self.selected_bodies(question):
            parts.append(
                f"--- Skill: {skill.name} ---\n{skill.body}\n--- end skill: {skill.name} ---"
            )

        return "\n\n".join(parts)
