"""
orchestrator/validator.py — Validation & Scoring Framework  (Phase 4)

Implements spec §7 in full:

  §7.2  Eight validation dimensions
  §7.3  Composite scoring model (weighted mean)
  §7.4  Per-task-type acceptance thresholds
  §7.6  Revision triggers
  §7.7  ValidationReport schema
  §7.9  Diminishing-returns detection

Phase 2 change: TaskValidator.validate() now loads the DepartmentContextProfile
for task.owner_department and uses its validation_weights instead of the global
_DEFAULT_WEIGHTS. This means:
  - A research task is judged primarily on accuracy + completeness.
  - An implementation task is judged primarily on structural validity + accuracy.
  - A red team task is judged primarily on risk_profile quality.
  - etc. (see llm/department_profiles.py for full weight tables)

_structural() gains a min_findings parameter so departments that must produce
many findings (QA, red_team) are held to a higher minimum than others.

Two evaluation modes
--------------------
HEURISTIC (always runs, zero latency):
    Structural checks — required fields, schema, min lengths.
    Completeness     — expected output types, content richness.
    Coherence        — vague-language detection, contradiction hints.
    Risk quality     — are risks mechanistic or just generic labels?
    Readiness        — will downstream tasks have something to act on?
    Usefulness       — does output add content beyond restating the task?
    Constraint check — does output reference constraint violations?
    Confidence gate  — detects obvious confidence inflation.

LLM-ENHANCED (opt-in via use_llm=True):
    Each task.validation_criteria item scored 0.0–1.0 by a validator
    LLM call. Scores feed into the accuracy dimension (60/40 blend with
    heuristic). Falls back to heuristic-only if LLM is unavailable.

Integration
-----------
Place this file at:  orchestrator/validator.py

Called from agent_runner.py after the AgentOutput is parsed.
ValidationReport is attached to RunResult and consumed by
engine_orchestrator._handle_result().

No new packages are required — all imports are stdlib or already in
requirements.txt (pydantic, structlog).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from llm.department_profiles import effective_weights, get_department_profile
from llm.schemas import AgentOutput
from models.task import Task, TaskType

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ValidationResult(StrEnum):
    PASS   = "pass"
    REVISE = "revise"   # fixable — retry with targeted feedback
    FAIL   = "fail"     # fundamental; don't retry blindly


# ---------------------------------------------------------------------------
# Dimension scores  (spec §7.2)
# ---------------------------------------------------------------------------


class DimensionScores(BaseModel):
    """Normalised 0.0–1.0 scores for each of the eight validation dimensions."""

    structural:            float = Field(default=1.0, ge=0.0, le=1.0)
    coherence:             float = Field(default=1.0, ge=0.0, le=1.0)
    completeness:          float = Field(default=0.5, ge=0.0, le=1.0)
    accuracy:              float = Field(default=0.5, ge=0.0, le=1.0)
    risk_profile:          float = Field(default=0.5, ge=0.0, le=1.0)
    readiness:             float = Field(default=0.5, ge=0.0, le=1.0)
    usefulness:            float = Field(default=0.5, ge=0.0, le=1.0)
    constraint_compliance: float = Field(default=1.0, ge=0.0, le=1.0)

    def composite(self, weights: dict[str, float] | None = None) -> float:
        w = weights or _DEFAULT_WEIGHTS
        total = sum(getattr(self, dim) * wt for dim, wt in w.items() if hasattr(self, dim))
        return round(min(1.0, max(0.0, total)), 4)


_DEFAULT_WEIGHTS: dict[str, float] = {
    "structural":            0.20,
    "coherence":             0.15,
    "completeness":          0.20,
    "accuracy":              0.20,
    "risk_profile":          0.08,
    "readiness":             0.10,
    "usefulness":            0.05,
    "constraint_compliance": 0.02,
}

# Per-task-type minimum thresholds  (spec §7.4)
# The validator takes max(task.quality_threshold, this value).
_TYPE_THRESHOLDS: dict[str, float] = {
    "research":       0.70,
    "design":         0.78,
    "implementation": 0.80,
    "critique":       0.72,
    "validation":     0.82,
    "merge":          0.75,
    "refinement":     0.76,
    "planning":       0.73,
    "documentation":  0.70,
}


def _threshold_for(task: Task) -> float:
    """
    Compute the effective validation threshold for a task.

    Applies settings.validation_score_scale so operators can dial thresholds
    up or down without editing code.  The scale defaults to 1.0 (no change).
    Use 0.65–0.75 when running small local models (Ollama 7B–13B) to prevent
    all tasks from failing.  Set back to 1.0 for cloud models (Claude, GPT-4).
    """
    from config import get_settings
    scale = max(0.3, min(1.0, get_settings().validation_score_scale))
    base = max(task.quality_threshold, _TYPE_THRESHOLDS.get(task.type.value, 0.75))
    return round(base * scale, 4)


# ---------------------------------------------------------------------------
# ValidationReport  (spec §7.7)
# ---------------------------------------------------------------------------


class ValidationReport(BaseModel):
    task_id:         str
    result:          ValidationResult
    scores:          DimensionScores
    composite_score: float = Field(ge=0.0, le=1.0)
    threshold:       float = Field(ge=0.0, le=1.0)

    issues:         list[str] = Field(default_factory=list,
        description="Specific problems; one sentence each.")
    required_fixes: list[str] = Field(default_factory=list,
        description="Concrete instructions for the retry agent.")

    validator_notes:  str = ""
    criteria_scores:  dict[str, float] = Field(default_factory=dict)
    validated_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms:       int = 0
    used_llm:         bool = False

    # Phase 2: record which department weights were applied
    department_weights_used: str = "default"

    def to_corrective_feedback(self) -> str:
        """Format for injection into AgentInputBundle.corrective_feedback."""
        parts: list[str] = [
            f"Validation score: {self.composite_score:.0%} "
            f"(threshold: {self.threshold:.0%}, result: {self.result.value})"
        ]
        if self.issues:
            parts.append("Issues:\n" + "\n".join(f"  - {i}" for i in self.issues))
        if self.required_fixes:
            parts.append("Required fixes:\n" +
                "\n".join(f"  {n+1}. {f}" for n, f in enumerate(self.required_fixes)))
        failing_criteria = {c: s for c, s in self.criteria_scores.items() if s < 0.6}
        if failing_criteria:
            parts.append("Criteria not met:\n" +
                "\n".join(f"  - {c}: {s:.0%}" for c, s in failing_criteria.items()))
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Heuristic dimension scorers
# Each returns: (score: float, issues: list[str], fixes: list[str])
# ---------------------------------------------------------------------------


def _structural(
    output: AgentOutput,
    min_findings: int = 2,
) -> tuple[float, list[str], list[str]]:
    """
    Phase 2 change: accepts min_findings parameter (sourced from
    DepartmentContextProfile.min_findings_expected) so departments
    like QA and red_team are held to a higher minimum than others.
    """
    issues, fixes, ded = [], [], 0.0

    if not output.summary or len(output.summary.strip()) < 30:
        issues.append("Summary missing or too short (< 30 chars).")
        fixes.append("Write a substantive one-paragraph summary of what you produced.")
        ded += 0.35

    if not output.findings:
        issues.append("No findings provided.")
        fixes.append(f"List at least {min_findings} discrete, defensible findings.")
        ded += 0.25
    elif len(output.findings) < min_findings:
        count = len(output.findings)
        issues.append(
            f"Only {count} finding{'s' if count != 1 else ''} provided — "
            f"minimum {min_findings} required for this department."
        )
        fixes.append(f"Expand to at least {min_findings} distinct findings.")
        ded += 0.10

    if output.status == "blocked" and not output.blocked_reason:
        issues.append("Status is 'blocked' but no blocked_reason given.")
        fixes.append("Explain exactly what you are blocked on.")
        ded += 0.30

    return round(max(0.0, 1.0 - ded), 3), issues, fixes


def _completeness(output: AgentOutput, task: Task) -> tuple[float, list[str], list[str]]:
    issues, fixes = [], []

    if task.expected_output_types:
        all_text = (output.summary + " ".join(output.findings)).lower()
        covered = sum(
            1 for t in task.expected_output_types
            if any(p in all_text for p in t.lower().replace("_", " ").split())
        )
        ratio = covered / max(1, len(task.expected_output_types))
        score = 0.4 + 0.6 * ratio
        if ratio < 0.5:
            missing = [t for t in task.expected_output_types
                       if not any(p in all_text for p in t.lower().replace("_", " ").split())]
            issues.append(f"Expected output types not clearly produced: {missing}.")
            fixes.append(f"Address each of: {', '.join(task.expected_output_types)}.")
    else:
        richness = min(1.0, (len(output.findings) + len(output.recommendations)) / 8.0)
        score = 0.4 + 0.6 * richness

    if len(output.open_questions) > 5:
        issues.append(f"{len(output.open_questions)} open questions remain — too many for a complete task.")
        fixes.append("Resolve or explicitly close out open questions that are not genuine blockers.")
        score = max(0.0, score - 0.15)

    return round(max(0.0, score), 3), issues, fixes


def _coherence(output: AgentOutput) -> tuple[float, list[str], list[str]]:
    issues, fixes = [], []
    score = 1.0
    all_text = " ".join(output.findings + output.recommendations + [output.summary])

    vague_patterns = [
        r"\bfurther research (is|may be) needed\b",
        r"\bit (could|might|may) be\b",
        r"\bvarious (factors|reasons|considerations)\b",
    ]
    hits = sum(1 for p in vague_patterns if re.search(p, all_text, re.IGNORECASE))
    if hits >= 2:
        score -= 0.12 * hits
        issues.append(f"Output contains {hits} instance(s) of vague language.")
        fixes.append("Replace hedging ('it might be', 'further research needed') with specific, defensible statements.")

    return round(max(0.0, min(1.0, score)), 3), issues, fixes


def _risk_profile(output: AgentOutput) -> tuple[float, list[str], list[str]]:
    if not output.risks:
        return 0.4, ["No risks identified."], [
            "Identify at least 2 risks. Name the failure mechanism and consequence, "
            "not just 'it might fail'."
        ]

    specific_patterns = [
        r"\bif\b.{5,50}\bthen\b", r"\bbecause\b", r"\bdue to\b",
        r"\b(percent|%|threshold|limit|deadline)\b",
    ]
    scores = []
    vague: list[str] = []
    for risk in output.risks:
        hits = sum(1 for p in specific_patterns if re.search(p, risk, re.IGNORECASE))
        if hits >= 1 and len(risk) >= 30:
            scores.append(0.9)
        elif len(risk) >= 20:
            scores.append(0.55)
        else:
            scores.append(0.2)
            vague.append(risk[:70])

    avg = sum(scores) / len(scores)
    issues, fixes = [], []
    if vague:
        issues.append(f"Vague risks (no failure mechanism): {vague[:2]}.")
        fixes.append(
            "For each risk state: (1) trigger condition, (2) failure mechanism, "
            "(3) downstream consequence."
        )
    return round(avg, 3), issues, fixes


def _readiness(output: AgentOutput, task: Task) -> tuple[float, list[str], list[str]]:
    issues, fixes = [], []
    research_types = {"research", "planning", "documentation"}
    action_types   = {"design", "implementation", "refinement", "merge"}

    if task.type.value in research_types:
        score = 0.5 + 0.35 * min(1.0, len(output.findings) / 4) + 0.15 * bool(output.next_actions)
    elif task.type.value in action_types:
        score = 0.5 + 0.25 * bool(output.recommendations) + 0.25 * bool(output.findings)
        if not output.recommendations:
            issues.append(f"A {task.type.value} task should include concrete recommendations.")
            fixes.append("Add at least 2 actionable recommendations.")
    else:
        score = 0.5 + 0.3 * bool(output.findings) + 0.2 * bool(output.next_actions or output.recommendations)

    if len(output.open_questions) >= 4:
        score -= 0.2
        issues.append(f"{len(output.open_questions)} unresolved open questions may block downstream tasks.")
        fixes.append("Resolve or explicitly dismiss non-blocking open questions.")

    return round(max(0.0, min(1.0, score)), 3), issues, fixes


def _usefulness(output: AgentOutput) -> tuple[float, list[str], list[str]]:
    content_len = len(output.summary) + sum(len(f) for f in output.findings)
    if content_len < 100:
        return 0.2, ["Output too short to be useful."], [
            "Minimum: 2-3 sentence summary and 3+ distinct findings."
        ]
    unique_words = {w.lower() for f in output.findings for w in f.split() if len(w) > 6}
    score = min(1.0, 0.4 + 0.04 * len(unique_words) + 0.05 * (len(output.findings) + len(output.recommendations)))
    return round(score, 3), [], []


def _constraint_compliance(output: AgentOutput, hard_constraints: list[str]) -> tuple[float, list[str], list[str]]:
    if not hard_constraints:
        return 1.0, [], []
    all_text = (output.summary + " ".join(output.findings + output.risks + output.assumptions)).lower()
    if re.search(r"\b(violat|ignor|bypass|overrid|circumvent|skip)\b", all_text, re.IGNORECASE):
        return 0.5, ["Output may reference violating a project constraint."], [
            "Review hard constraints and confirm your output respects all of them."
        ]
    return 1.0, [], []


def _confidence_check(output: AgentOutput, structural: float, completeness: float) -> tuple[float, list[str], list[str]]:
    """Returns a composite adjustment (negative = penalty)."""
    floor = min(structural, completeness)
    if output.confidence > 0.85 and floor < 0.55:
        return -0.10, [
            f"Confidence ({output.confidence:.0%}) appears inflated vs "
            f"structural/completeness scores ({floor:.0%})."
        ], ["Calibrate confidence honestly."]
    return 0.0, [], []


# ---------------------------------------------------------------------------
# LLM-enhanced criterion scoring  (spec §7.5)
# ---------------------------------------------------------------------------


async def _llm_criteria_score(
    output: AgentOutput,
    task: Task,
    timeout_s: float = 45.0,
) -> tuple[dict[str, float], str, float]:
    """Returns (criteria_scores, validator_notes, accuracy_override)."""
    if not task.validation_criteria:
        return {}, "", 0.5
    try:
        from llm.client import get_client
        client = get_client()
    except Exception:
        return {}, "", 0.5

    criteria_block = "\n".join(f'  {i+1}. "{c}"' for i, c in enumerate(task.validation_criteria))
    output_block = (
        f"Summary: {output.summary}\n\n"
        f"Findings:\n" + "\n".join(f"  - {f}" for f in output.findings[:10]) + "\n\n"
        f"Recommendations:\n" + "\n".join(f"  - {r}" for r in output.recommendations[:6])
    )
    from llm.schemas import Message
    system = (
        "You are a strict validation agent. Score each criterion 0.0 (not met) "
        "to 1.0 (fully met). Be objective. Respond ONLY with valid JSON."
    )
    user = (
        f"CRITERIA:\n{criteria_block}\n\n"
        f"OUTPUT:\n{output_block[:2000]}\n\n"
        f"Return JSON:\n"
        '{{"criteria_scores": {{"<criterion>": <float>, ...}}, '
        '"validator_notes": "<2-3 sentence summary>"}}'
    )
    try:
        raw, _ = await asyncio.wait_for(
            client.chat_json(
                [Message(role="system", content=system), Message(role="user", content=user)],
                max_tokens=1024, retry_limit=1,
            ),
            timeout=timeout_s,
        )
        raw_scores: dict[str, Any] = raw.get("criteria_scores", {})
        notes: str = raw.get("validator_notes", "")
        merged: dict[str, float] = {}
        score_vals = list(raw_scores.values())
        for i, crit in enumerate(task.validation_criteria):
            if crit in raw_scores:
                merged[crit] = max(0.0, min(1.0, float(raw_scores[crit])))
            elif i < len(score_vals):
                merged[crit] = max(0.0, min(1.0, float(score_vals[i])))
            else:
                merged[crit] = 0.5
        accuracy = sum(merged.values()) / len(merged) if merged else 0.5
        return merged, notes, accuracy
    except Exception as exc:
        log.warning("validator.llm_criteria_failed", error=str(exc))
        return {}, "", 0.5


# ---------------------------------------------------------------------------
# Diminishing-returns detection  (spec §7.9)
# ---------------------------------------------------------------------------


@dataclass
class _IterHistory:
    scores: list[float] = field(default_factory=list)

    def add(self, s: float) -> None:
        self.scores.append(round(s, 4))

    def plateauing(self, window: int = 3, min_gain: float = 0.02) -> bool:
        if len(self.scores) < window:
            return False
        r = self.scores[-window:]
        return (max(r) - min(r)) < min_gain

    def degrading(self) -> bool:
        return len(self.scores) >= 2 and self.scores[-1] < self.scores[-2]


_iter_registry: dict[str, _IterHistory] = {}


def _get_history(task_id: str) -> _IterHistory:
    if task_id not in _iter_registry:
        _iter_registry[task_id] = _IterHistory()
    return _iter_registry[task_id]


def clear_iteration_history(task_id: str) -> None:
    """Call when a task is approved so history doesn't carry over to re-runs."""
    _iter_registry.pop(task_id, None)


# ---------------------------------------------------------------------------
# TaskValidator
# ---------------------------------------------------------------------------


class TaskValidator:

    async def validate(
        self,
        output: AgentOutput,
        task: Task,
        hard_constraints: list[str] | None = None,
        use_llm: bool = False,
    ) -> ValidationReport:
        t0 = time.monotonic()
        hard_constraints = hard_constraints or []

        # Phase 2: load department profile for specialised scoring
        profile = get_department_profile(task.owner_department.value)
        dept_weights = effective_weights(profile)
        min_findings = profile.min_findings_expected

        # Heuristic pass — use department's min_findings in structural check
        s_score,  s_iss,  s_fix  = _structural(output, min_findings=min_findings)
        co_score, co_iss, co_fix = _coherence(output)
        c_score,  c_iss,  c_fix  = _completeness(output, task)
        r_score,  r_iss,  r_fix  = _risk_profile(output)
        rd_score, rd_iss, rd_fix = _readiness(output, task)
        u_score,  u_iss,  u_fix  = _usefulness(output)
        cn_score, cn_iss, cn_fix = _constraint_compliance(output, hard_constraints)

        heuristic_accuracy = output.confidence * 0.5 + s_score * 0.25 + c_score * 0.25
        conf_adj, cf_iss, cf_fix = _confidence_check(output, s_score, c_score)
        heuristic_accuracy = max(0.0, heuristic_accuracy + conf_adj)

        # LLM pass (optional)
        criteria_scores: dict[str, float] = {}
        validator_notes = ""
        used_llm = False
        accuracy = heuristic_accuracy

        if use_llm and task.validation_criteria:
            try:
                criteria_scores, validator_notes, llm_acc = await _llm_criteria_score(output, task)
                if criteria_scores:
                    accuracy = 0.6 * llm_acc + 0.4 * heuristic_accuracy
                    used_llm = True
            except Exception as exc:
                log.warning("validator.llm_error", error=str(exc))

        scores = DimensionScores(
            structural=s_score,
            coherence=co_score,
            completeness=c_score,
            accuracy=round(accuracy, 3),
            risk_profile=r_score,
            readiness=rd_score,
            usefulness=u_score,
            constraint_compliance=cn_score,
        )

        # Phase 2: use department-specific weights instead of global defaults
        composite = scores.composite(dept_weights)

        # Diminishing-returns tracking
        hist = _get_history(task.task_id)
        hist.add(composite)
        plateau  = hist.plateauing()
        degraded = hist.degrading()

        all_issues = s_iss + c_iss + co_iss + r_iss + rd_iss + u_iss + cn_iss + cf_iss
        all_fixes  = s_fix + c_fix + co_fix + r_fix + rd_fix + u_fix + cn_fix + cf_fix

        threshold = _threshold_for(task)
        result = self._decide(composite, threshold, output, scores, plateau, degraded)

        if plateau and result == ValidationResult.REVISE:
            all_issues.append("Scores are plateauing — current strategy is not converging.")
            all_fixes.append("Significantly change your approach; the current method is stuck.")
        elif degraded:
            all_issues.append("Score degraded from previous attempt.")
            all_fixes.append("Revert to what worked in the prior attempt and iterate from there.")

        latency_ms = int((time.monotonic() - t0) * 1000)

        if not validator_notes:
            low = [f"{d} ({getattr(scores, d):.0%})"
                   for d in ("structural","completeness","accuracy","coherence","readiness")
                   if getattr(scores, d) < 0.55]
            validator_notes = (
                f"Composite {composite:.0%} vs threshold {threshold:.0%} "
                f"[{profile.role} weights]. "
                + (f"Lowest: {', '.join(low)}." if low else "All dimensions acceptable.")
            )

        log.info(
            "validator.complete",
            task_id=task.task_id,
            department=task.owner_department.value,
            result=result.value,
            composite=composite,
            threshold=threshold,
            used_llm=used_llm,
            issues=len(all_issues),
            latency_ms=latency_ms,
        )

        # ── Diagnostic breakdown on failure (makes debugging much easier) ────
        if result != ValidationResult.PASS:
            dim_scores = {
                "structural":    round(s_score, 2),
                "completeness":  round(c_score, 2),
                "coherence":     round(co_score, 2),
                "accuracy":      round(accuracy, 2),
                "risk_profile":  round(r_score, 2),
                "readiness":     round(rd_score, 2),
                "usefulness":    round(u_score, 2),
            }
            failing = {k: v for k, v in dim_scores.items() if v < threshold}
            log.warning(
                "validator.fail_breakdown",
                task_id=task.task_id,
                task_type=task.type.value,
                composite=composite,
                threshold=threshold,
                failing_dimensions=failing,
                top_issue=all_issues[0] if all_issues else "none",
                hint=(
                    "If ALL tasks fail like this, lower VALIDATION_SCORE_SCALE in .env "
                    f"(current: {threshold / max(task.quality_threshold, 0.01):.2f}) "
                    "or switch to a stronger LLM provider (LLM_PROVIDER=anthropic)."
                ) if composite < 0.55 else "",
            )

        return ValidationReport(
            task_id=task.task_id,
            result=result,
            scores=scores,
            composite_score=composite,
            threshold=threshold,
            issues=all_issues[:12],
            required_fixes=all_fixes[:8],
            validator_notes=validator_notes,
            criteria_scores=criteria_scores,
            latency_ms=latency_ms,
            used_llm=used_llm,
            department_weights_used=profile.role,
        )

    @staticmethod
    def _decide(
        composite: float,
        threshold: float,
        output: AgentOutput,
        scores: DimensionScores,
        plateau: bool,
        degraded: bool,
    ) -> ValidationResult:
        # Hard-fail conditions
        if scores.structural <= 0.20:
            return ValidationResult.FAIL
        if output.status == "blocked" and not output.blocked_reason:
            return ValidationResult.FAIL
        if plateau and composite < threshold - 0.15:
            return ValidationResult.FAIL
        # Pass
        if composite >= threshold and output.status != "blocked":
            return ValidationResult.PASS
        return ValidationResult.REVISE


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_validator: TaskValidator | None = None


def get_validator() -> TaskValidator:
    global _validator
    if _validator is None:
        _validator = TaskValidator()
    return _validator


async def validate_and_build_feedback(
    output: AgentOutput,
    task: Task,
    hard_constraints: list[str] | None = None,
    use_llm: bool = False,
) -> tuple[ValidationReport, str]:
    """Convenience wrapper → (ValidationReport, corrective_feedback_string).

    Phase 7: fires an instability checkpoint when the validator detects a
    score plateau (same task failing to improve across 3+ attempts).
    """
    report = await get_validator().validate(output, task, hard_constraints, use_llm=use_llm)
    feedback = report.to_corrective_feedback() if report.result != ValidationResult.PASS else ""

    # Checkpoint on instability — plateau means the project state is worth preserving
    # in case the operator needs to inspect or redirect.
    hist = _get_history(task.task_id)
    if hist.plateauing():
        try:
            from orchestrator.checkpoint_manager import (
                CheckpointTrigger,
                get_checkpoint_manager,
            )
            await get_checkpoint_manager().save(
                task.project_id, trigger=CheckpointTrigger.INSTABILITY
            )
        except Exception as exc:
            log.warning(
                "validator.checkpoint_failed",
                task_id=task.task_id,
                error=str(exc),
            )

        # Phase 9: plateau is a strong learning signal — record what the
        # repeated corrective strategy was and that it didn't converge.
        try:
            from memory.manager import get_memory_manager
            _mgr = get_memory_manager()
            await _mgr.submit_candidates(
                candidates=[{
                    "category": "failure_reason",
                    "title": f"Strategy plateau: {task.owner_department.value}/{task.type.value}",
                    "content": (
                        f"Validation score plateaued for task type "
                        f"'{task.type.value}' in '{task.owner_department.value}'. "
                        f"Score stuck near {report.composite_score:.0%} vs "
                        f"threshold {report.threshold:.0%}. "
                        f"Top issues: {'; '.join(report.issues[:2])}"
                    ),
                    "confidence": 0.80,
                    "tags": [
                        "plateau", "strategy_failure",
                        task.owner_department.value, task.type.value,
                    ],
                }],
                project_id=task.project_id,
                task_id=task.task_id,
                agent_role="learning",
            )
        except Exception:
            pass

    return report, feedback
