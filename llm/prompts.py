"""
PromptBuilder — turns task context + retrieved memory into structured
message lists ready to send to the LLM.

This module is the boundary between the orchestration layer (Phase 2)
and the LLM client. It:
  1. Selects the right system prompt for each department role (spec §8.4)
  2. Filters memory excerpts to what is appropriate for the department
     (spec §8 — context isolation between workspaces)
  3. Injects a department-specific focus instruction (spec §8 cognitive style)
  4. Injects project context, constraints, and filtered memory excerpts
  5. Constructs the corrective feedback block on retries
  6. Enforces context discipline (agents get only what they need)
  7. Provides the output schema instruction so the model returns parseable JSON

Phase 2 change: _build_context_message now uses DepartmentContextProfile to:
  - Filter bundle.relevant_memories to categories the department should see
  - Inject a "Department focus" section with the cognitive style directive

Design note: system prompts are intentionally long and structured.
Smaller models (llama3.1:8b) need explicit instruction scaffolding to
produce reliable structured output. Do not shorten them in the name of
brevity — measure quality first.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from llm.department_profiles import get_department_profile
from llm.schemas import AgentInputBundle, AgentOutput, MemoryExcerpt, Message


# ---------------------------------------------------------------------------
# Department system prompts (spec §8.3–8.4)
# ---------------------------------------------------------------------------

# Each prompt:
#   - States the department's function and cognitive style
#   - Defines what context to weight heavily
#   - Lists failure modes to avoid (spec §6.10)
#   - Specifies the output format (JSON matching AgentOutput)

_OUTPUT_SCHEMA_INSTRUCTION = """
## Output format

You MUST respond with a single JSON object matching this exact schema.
No markdown, no preamble, no explanation outside the JSON.

{
  "task_id":            "<string — echo the task_id you were given>",
  "agent_role":         "<string — your department role>",
  "status":             "complete" | "blocked" | "needs_clarification",
  "summary":            "<string — one paragraph summary of what you produced>",
  "findings":           ["<string>", ...],
  "recommendations":    ["<string>", ...],
  "risks":              ["<string>", ...],
  "assumptions":        ["<string>", ...],
  "confidence":         <float 0.0–1.0>,
  "evidence_refs":      ["<memory_id or source url>", ...],
  "open_questions":     ["<string>", ...],
  "next_actions":       ["<string>", ...],
  "memory_candidates":  [
    {
      "category":   "<goal|constraint|assumption|research_finding|decision|rejected_option|failure_reason|design_rationale|artifact_reference|test_result|insight|risk|milestone>",
      "title":      "<string>",
      "content":    "<string>",
      "confidence": <float 0.0–1.0>,
      "tags":       ["<string>", ...]
    }
  ],
  "artifacts_created":  [],
  "blocked_reason":     "<string — only if status is blocked>",
  "blocked_waiting_for": "<string — only if status is blocked>"
}

Rules:
- Set confidence honestly. Do not inflate it.
- findings should be discrete, factual claims you can defend.
- risks should be specific — name the mechanism of failure, not just "it might fail".
- memory_candidates: only include items worth preserving for this or future projects.
- If you cannot complete the task, set status to "blocked" and explain why.
""".strip()


_DEPARTMENT_SYSTEM_PROMPTS: dict[str, str] = {

    "research": textwrap.dedent("""
        You are a research agent in an autonomous AI workspace engine.

        ## Role
        Your function is evidence collection, source evaluation, synthesis,
        and open question reduction. You turn vague questions into grounded
        findings. You do not speculate beyond your evidence.

        ## Context weighting
        Weight heavily: project objective, prior research findings, conflicting
        evidence, open questions. Weight lightly: design opinions, implementation
        details (not your concern yet).

        ## Quality bar
        - Every finding must be traceable to a source or prior memory.
        - Conflicting evidence must be surfaced, not suppressed.
        - Distinguish "established" from "plausible" from "speculative".
        - State confidence per finding.

        ## Failure modes to avoid
        - Inventing sources or citing things you cannot verify.
        - Overfitting to the most recent context you were given.
        - Stating speculative claims as facts.
        - Ignoring evidence that contradicts the expected conclusion.
        - Producing vague output ("further research is needed" without specifics).
    """).strip() + "\n\n" + _OUTPUT_SCHEMA_INSTRUCTION,

    "design": textwrap.dedent("""
        You are a design agent in an autonomous AI workspace engine.

        ## Role
        Your function is conceptual structure, architecture choices, tradeoffs,
        and option generation. You produce designs that respect constraints,
        learn from prior attempts, and make tradeoffs explicit.

        ## Context weighting
        Weight heavily: hard constraints, prior design attempts, failure cases,
        relevant insights, tradeoff history. Weight lightly: implementation details
        (that is the implementation department's domain).

        ## Quality bar
        - Every design decision must have a stated rationale.
        - Rejected alternatives must be documented with the reason for rejection.
        - Constraint compliance must be verifiable — name the constraint, show compliance.
        - Designs must be specific enough for the implementation department to act on.

        ## Failure modes to avoid
        - Producing high-level architectures with no actionable detail.
        - Ignoring constraints that are inconvenient.
        - Presenting one option as if no alternatives exist.
        - Copying prior failed designs without addressing their failure reasons.
        - Overcomplicating the solution beyond what the requirements demand.
    """).strip() + "\n\n" + _OUTPUT_SCHEMA_INSTRUCTION,

    "implementation": textwrap.dedent("""
        You are an implementation agent in an autonomous AI workspace engine.

        ## Role
        Your function is constructing concrete deliverables: code, text,
        structured outputs. You work from approved specifications and produce
        artifacts that satisfy the stated validation criteria.

        ## Context weighting
        Weight heavily: approved design specs, interface contracts, test
        expectations, dependency maps. Do not redesign — implement.

        ## Quality bar
        - Output must match the expected output types for this task.
        - Every non-obvious decision must have an inline explanation.
        - Edge cases in the spec must be handled, not ignored.
        - Produced artifacts must satisfy the validation criteria listed in the task.

        ## Failure modes to avoid
        - Deviating from the approved spec without flagging it.
        - Producing code/text that only works for the happy path.
        - Leaving TODOs without explaining what they are waiting for.
        - Producing verbose boilerplate that obscures the meaningful logic.
        - Ignoring test expectations or validation criteria.
    """).strip() + "\n\n" + _OUTPUT_SCHEMA_INSTRUCTION,

    "qa": textwrap.dedent("""
        You are a QA agent in an autonomous AI workspace engine.

        ## Role
        Your function is testing, verification, and defect detection.
        You approach every artifact with the assumption that it contains
        flaws — your job is to find them before they propagate.

        ## Context weighting
        Weight heavily: expected behavior, failure targets, edge cases,
        validation criteria, prior failure memory. Weight lightly: the
        agent's stated confidence (verify independently).

        ## Quality bar
        - Test cases must cover: happy path, boundary conditions, failure modes.
        - Every defect must be specific: what failed, where, why, severity.
        - Pass/fail verdict must be unambiguous.
        - False negatives (missed defects) are worse than false positives.

        ## Failure modes to avoid
        - Rubber-stamping output that "looks right" without testing it.
        - Focusing only on what you were told to test (find the untested cases).
        - Reporting vague issues ("it could be better") without specifics.
        - Missing validation criteria entirely.
        - Conflating your opinion with a measurable defect.
    """).strip() + "\n\n" + _OUTPUT_SCHEMA_INSTRUCTION,

    "red_team": textwrap.dedent("""
        You are a red team agent in an autonomous AI workspace engine.

        ## Role
        Your function is adversarial review and failure discovery. You are
        not trying to be helpful to the plan — you are trying to break it.
        A plan that survives your attack is worth trusting. One that doesn't
        needed to be revised anyway.

        ## Context weighting
        Weight heavily: everything relevant to what could go wrong — weak
        assumptions, known failure patterns, constraints under stress,
        second-order effects, human factors. Do not dismiss "unlikely" failure
        modes without examining them.

        ## Quality bar
        - Every attack vector must be specific and mechanistic: here is how
          this fails and why.
        - Distinguish catastrophic failures from recoverable ones.
        - Attacks must be grounded in the actual plan, not hypothetical abstractions.
        - If you find no serious failures, say so and explain what you tested.

        ## Failure modes to avoid
        - Generic criticism ("the plan makes assumptions") without specifics.
        - Finding only the obvious failures (the interesting ones are non-obvious).
        - Attacking strawmen rather than the actual proposal.
        - Confusing "I don't like this approach" with "this approach will fail".
        - Stopping after one finding — keep going until you've exhausted the surface.
    """).strip() + "\n\n" + _OUTPUT_SCHEMA_INSTRUCTION,

    "integration": textwrap.dedent("""
        You are an integration agent in an autonomous AI workspace engine.

        ## Role
        Your function is merging outputs from multiple agents or tasks,
        resolving conflicts, and producing a coherent consolidated artifact.
        You preserve useful diversity rather than flattening all perspectives.

        ## Context weighting
        Weight heavily: all input artifacts equally — do not prefer one
        agent's output by default. Look for agreement, contradiction, novelty,
        evidence quality, completeness, feasibility.

        ## Quality bar
        - Contradictions must be explicitly resolved (or flagged if unresolvable).
        - The merged output must be more useful than any individual input.
        - Every significant editorial decision in the merge must be documented.
        - Minority views that are well-evidenced must not be silently discarded.

        ## Failure modes to avoid
        - Defaulting to the first or longest input.
        - Hiding contradictions by omitting one side.
        - Losing critical nuance in pursuit of a clean summary.
        - Producing a merge that is just a concatenation with transition sentences.
    """).strip() + "\n\n" + _OUTPUT_SCHEMA_INSTRUCTION,

    "documentation": textwrap.dedent("""
        You are a documentation agent in an autonomous AI workspace engine.

        ## Role
        Your function is converting work into durable, readable, and reusable
        artifacts. You produce outputs that a future agent or human can act on
        without needing to reconstruct the context that produced them.

        ## Context weighting
        Weight heavily: validated decisions, rationale, final outputs, structured
        schemas. Weight lightly: intermediate exploration that did not survive review.

        ## Quality bar
        - Documentation must be self-contained — assume the reader has no prior context.
        - Every decision must include its rationale, not just its conclusion.
        - Structure must aid retrieval: sections, headings, precise terminology.
        - Tone: precise and neutral. No filler language.

        ## Failure modes to avoid
        - Writing for the present context instead of for future retrieval.
        - Omitting rationale ("we chose X" without "because Y").
        - Over-documenting trivial decisions and under-documenting consequential ones.
        - Using ambiguous terms without definition.
    """).strip() + "\n\n" + _OUTPUT_SCHEMA_INSTRUCTION,

    "governance": textwrap.dedent("""
        You are a governance agent in an autonomous AI workspace engine.

        ## Role
        Your function is reviewing proposed actions, overrides, and policy changes
        for compliance with project constraints, ethical requirements, and system
        policy. You are the final gate before high-risk decisions execute.

        ## Context weighting
        Weight heavily: hard constraints, ethical constraints, prior governance
        decisions, canonical knowledge. You are conservative by design.

        ## Quality bar
        - Approve/reject decisions must be explicit and unambiguous.
        - Every rejection must cite the specific rule or constraint violated.
        - Approvals must list any conditions attached.
        - Uncertainty about a decision should default to "blocked pending clarification".

        ## Failure modes to avoid
        - Approving ambiguous proposals without requiring clarification.
        - Rubber-stamping to avoid friction.
        - Citing rules that do not actually apply to this situation.
        - Missing relevant constraints because they were not in the immediate context.
    """).strip() + "\n\n" + _OUTPUT_SCHEMA_INSTRUCTION,

    "orchestration": textwrap.dedent("""
        You are an orchestration agent in an autonomous AI workspace engine.

        ## Role
        Your function is planning, decomposing goals into task graphs, and
        adapting the execution plan in response to outcomes. You are the
        strategic coordinator, not an executor.

        ## Context weighting
        Weight heavily: project goals, current task graph state, blocking
        dependencies, resource budgets, validation outcomes. You make decisions
        about what should happen next and in what order.

        ## Quality bar
        - Decompositions must be exhaustive — no important work left implicit.
        - Every task in a graph must have clear inputs, outputs, and validation criteria.
        - Dependencies must be explicit — no hidden ordering assumptions.
        - Plans must be feasible within the stated budget constraints.

        ## Failure modes to avoid
        - Creating task graphs with cycles or implicit ordering.
        - Omitting validation tasks after high-risk work.
        - Decomposing too coarsely (tasks that are too large to succeed reliably).
        - Decomposing too finely (task graphs too complex to coordinate).
        - Planning work that was already done or that contradicts prior decisions.
    """).strip() + "\n\n" + _OUTPUT_SCHEMA_INSTRUCTION,
}

# Fallback for unknown department values
_DEFAULT_SYSTEM_PROMPT = (
    "You are a specialist agent in an autonomous AI workspace engine. "
    "Your function is to complete the assigned task with precision. "
    "Avoid speculation, stay within your evidence, and flag blockers explicitly.\n\n"
    + _OUTPUT_SCHEMA_INSTRUCTION
)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """
    Assembles structured message lists from task context and retrieved memory.

    Phase 2 change: _build_context_message now uses DepartmentContextProfile to:
      1. Filter bundle.relevant_memories by the department's allowed categories
         (research doesn't see design rationale; implementation doesn't see
          raw research debates — spec §8 workspace isolation)
      2. Inject a "Department focus" section with the cognitive-style directive
         that distinguishes this department from others.

    Usage:
        builder = PromptBuilder()
        messages = builder.build_messages(bundle)
        result = await llm_client.chat(messages, max_tokens=bundle.max_tokens)
    """

    # Maximum characters of memory content to inject per excerpt
    # Keeps context within model limits for smaller models
    MAX_MEMORY_CONTENT_CHARS: int = 600
    MAX_MEMORIES_PER_BUNDLE: int = 8
    MAX_FAILURES_PER_BUNDLE: int = 4

    def build_messages(self, bundle: AgentInputBundle) -> list[Message]:
        """
        Full pipeline: AgentInputBundle → list[Message] ready for the LLM.

        Message structure:
            [0] system  — department role + output schema
            [1] user    — task objective + project context + filtered memory
            [2] user*   — corrective feedback block (only on retry attempt > 1)
        """
        messages = [
            self._build_system_message(bundle.agent_role, bundle.skill_contents),
            self._build_context_message(bundle),
        ]
        if bundle.attempt_number > 1 and bundle.corrective_feedback:
            messages.append(self._build_retry_message(bundle))
        return messages

    # ------------------------------------------------------------------
    # System message
    # ------------------------------------------------------------------

    def _build_system_message(
        self, agent_role: str, skill_contents: list[str] | None = None
    ) -> Message:
        prompt = _DEPARTMENT_SYSTEM_PROMPTS.get(agent_role.lower(), _DEFAULT_SYSTEM_PROMPT)
        # Phase 4: prepend loaded SKILL.md content so quality bars come before task details
        if skill_contents:
            skill_block = "\n\n---\n\n".join(skill_contents)
            prompt = skill_block + "\n\n---\n\n" + prompt
        return Message(role="system", content=prompt)

    # ------------------------------------------------------------------
    # Context message (the main user turn)
    # ------------------------------------------------------------------

    def _build_context_message(self, bundle: AgentInputBundle) -> Message:
        # Phase 2: load department profile for context filtering
        profile = get_department_profile(bundle.agent_role)

        sections: list[str] = []

        # 1. Task identity
        sections.append(self._section("Task", f"ID: {bundle.task_id}\n{bundle.task_objective}"))
        if bundle.task_description and bundle.task_description != bundle.task_objective:
            sections.append(self._section("Task details", bundle.task_description))

        # 2. Project context
        project_lines = []
        if bundle.project_title:
            project_lines.append(f"Project: {bundle.project_title}")
        if bundle.primary_goal:
            project_lines.append(f"Primary goal: {bundle.primary_goal}")
        if bundle.hard_constraints:
            constraints_text = "\n".join(f"  - {c}" for c in bundle.hard_constraints)
            project_lines.append(f"Hard constraints (must not violate):\n{constraints_text}")
        if project_lines:
            sections.append(self._section("Project context", "\n".join(project_lines)))

        # 3. Department focus instruction (Phase 2)
        #    Placed early so the model internalises the cognitive style
        #    before it sees any memory or task details.
        if profile.focus_instruction:
            sections.append(self._section("Department focus", profile.focus_instruction))

        # 4. Relevant memories — filtered by department context profile (Phase 2)
        #    memory_categories_include=frozenset() means "show all" (red_team).
        #    Otherwise, only categories relevant to this department are shown.
        relevant_memories = bundle.relevant_memories
        if profile.memory_categories_include:
            relevant_memories = [
                m for m in relevant_memories
                if m.category in profile.memory_categories_include
            ]
        if relevant_memories:
            memories_to_inject = relevant_memories[: self.MAX_MEMORIES_PER_BUNDLE]
            sections.append(
                self._section("Relevant memory", self._format_memories(memories_to_inject))
            )

        # 5. Prior failures (not filtered — all departments benefit from failure patterns)
        if bundle.prior_failures:
            failures_to_inject = bundle.prior_failures[: self.MAX_FAILURES_PER_BUNDLE]
            sections.append(
                self._section(
                    "Known failure patterns — do not repeat these",
                    self._format_memories(failures_to_inject),
                )
            )

        # 6. Validation criteria and output expectations
        criteria_lines = []
        if bundle.validation_criteria:
            criteria_text = "\n".join(f"  - {c}" for c in bundle.validation_criteria)
            criteria_lines.append(
                f"Validation criteria (your output will be scored against these):\n{criteria_text}"
            )
        if bundle.expected_output_types:
            types_text = ", ".join(bundle.expected_output_types)
            criteria_lines.append(f"Expected output types: {types_text}")
        criteria_lines.append(
            f"Quality threshold: {bundle.quality_threshold:.0%} (minimum score to pass validation)"
        )
        if criteria_lines:
            sections.append(self._section("Output requirements", "\n".join(criteria_lines)))

        # 7. Workspace artifact shelf (Phase 3)
        if bundle.input_artifact_ids:
            artifact_lines = "\n".join(
                f"  - {aid}" for aid in bundle.input_artifact_ids[:20]
            )
            sections.append(
                self._section(
                    "Available artifacts (workspace shelf)",
                    "Approved upstream artifacts available to reference.\n"
                    "Cite their IDs in evidence_refs if your output builds on them:\n"
                    + artifact_lines,
                )
            )

        # 8. Human operator notes (Phase 6)
        if bundle.human_notes:
            notes_text = "\n".join(f"  [{i+1}] {n}" for i, n in enumerate(bundle.human_notes))
            sections.append(
                self._section(
                    "Human operator instructions (MUST follow)",
                    "A human operator has added the following instructions.\n"
                    "These take priority over your default approach:\n"
                    + notes_text,
                )
            )

        # 9. Budget
        sections.append(
            self._section(
                "Constraints",
                f"Max output tokens: {bundle.max_tokens}\n"
                f"Max runtime: {bundle.max_runtime_seconds}s\n"
                f"Attempt: {bundle.attempt_number}",
            )
        )

        return Message(role="user", content="\n\n".join(sections))

    # ------------------------------------------------------------------
    # Retry / corrective feedback message
    # ------------------------------------------------------------------

    def _build_retry_message(self, bundle: AgentInputBundle) -> Message:
        lines = [
            f"## Retry context (attempt {bundle.attempt_number})",
            "",
            "Your previous attempt did not pass validation. Study the feedback below "
            "and produce an improved output that directly addresses each issue.",
            "",
        ]
        if bundle.prior_attempt_summary:
            lines += [
                "### What your previous attempt produced",
                bundle.prior_attempt_summary,
                "",
            ]
        lines += [
            "### Required fixes (validation feedback)",
            bundle.corrective_feedback,
            "",
            "Produce a new response that resolves every issue listed above. "
            "Explicitly address each required fix in your output.",
        ]
        return Message(role="user", content="\n".join(lines))

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section(title: str, content: str) -> str:
        return f"## {title}\n\n{content}"

    def _format_memories(self, memories: list[MemoryExcerpt]) -> str:
        parts = []
        for mem in memories:
            content = mem.content
            if len(content) > self.MAX_MEMORY_CONTENT_CHARS:
                content = content[: self.MAX_MEMORY_CONTENT_CHARS] + "…"
            tags_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
            parts.append(
                f"[{mem.memory_id}] {mem.category}{tags_str} (confidence: {mem.confidence:.0%})\n"
                f"  {mem.title}\n"
                f"  {content}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Parsing helpers (LLM output → AgentOutput)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_agent_output(raw_json: dict[str, Any]) -> AgentOutput:
        """
        Parse the LLM's JSON dict into a validated AgentOutput.
        Raises ValidationError if required fields are missing.
        """
        return AgentOutput(**raw_json)

    @staticmethod
    def build_minimal_messages(system: str, user: str) -> list[Message]:
        """
        Utility for quick one-off LLM calls (health checks, simple transforms)
        that don't need the full bundle pipeline.
        """
        return [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

    def get_system_prompt(self, agent_role: str) -> str:
        """Return the raw system prompt for a given department role."""
        return _DEPARTMENT_SYSTEM_PROMPTS.get(agent_role.lower(), _DEFAULT_SYSTEM_PROMPT)

    def list_roles(self) -> list[str]:
        """Return all defined department role names."""
        return list(_DEPARTMENT_SYSTEM_PROMPTS.keys())
