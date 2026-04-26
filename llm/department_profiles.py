"""
llm/department_profiles.py — Per-department context and validation profiles.

Spec §8 (Agent Workspace Model) defines that each department sees a
deliberately different slice of project context:

  Research    → evidence, open questions, prior findings
  Design      → constraints, approved requirements, tradeoff history
  Implementation → approved spec, interface contracts (NOT research debates)
  QA          → expected behaviour, failure patterns, test results
  Red Team    → everything, with emphasis on weak points
  Integration → all upstream artifacts being merged
  Documentation → validated decisions and rationale
  Governance  → constraints, policies, canonical knowledge
  Orchestration → goal state, graph state, risk landscape

This module defines:
  DepartmentContextProfile  — dataclass per department
  DEPARTMENT_PROFILES       — registry dict keyed by DepartmentOwner.value
  get_department_profile()  — safe lookup (never raises)
  effective_weights()       — builds the normalised validation weight dict
                               for a profile (used by TaskValidator)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Default validation dimension weights (spec §7.2)
# Reproduced here to avoid a circular import with validator.py
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "structural":             0.20,
    "coherence":              0.15,
    "completeness":           0.20,
    "accuracy":               0.20,
    "risk_profile":           0.08,
    "readiness":              0.10,
    "usefulness":             0.05,
    "constraint_compliance":  0.02,
}


# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepartmentContextProfile:
    """
    Describes what a department needs to see and how its outputs are judged.

    Attributes
    ----------
    role
        The DepartmentOwner enum value this profile covers.
    memory_categories_include
        Which memory categories are shown in the context bundle.
        An empty frozenset means "show everything" (used by red_team).
    focus_instruction
        A short directive injected into the agent's context as a
        "Department focus" section. Reinforces the cognitive style
        that distinguishes this department from others.
    validation_weights
        Full 8-dimension weight dict. Must sum to 1.0. These replace
        _DEFAULT_WEIGHTS for outputs from this department. Each department
        weights the dimensions that matter most for its output type.
    min_findings_expected
        Minimum number of findings the structural validator requires.
        QA and red_team need more findings than documentation.
    """

    role: str
    memory_categories_include: frozenset[str]  # empty = all
    focus_instruction: str
    validation_weights: dict[str, float]
    min_findings_expected: int = 2


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

DEPARTMENT_PROFILES: dict[str, DepartmentContextProfile] = {

    "research": DepartmentContextProfile(
        role="research",
        memory_categories_include=frozenset({
            "goal", "constraint", "assumption",
            "research_finding", "insight", "risk",
        }),
        focus_instruction=(
            "Your output is evidence, not opinion. Every finding must be traceable. "
            "Distinguish 'established' from 'plausible' from 'speculative' — explicitly. "
            "Surface conflicting evidence; do not suppress it to reach a clean conclusion."
        ),
        validation_weights={
            # Accuracy and completeness are the core value of research
            "structural":            0.15,
            "coherence":             0.18,
            "completeness":          0.20,
            "accuracy":              0.25,  # ↑ highest — evidence quality is the point
            "risk_profile":          0.06,
            "readiness":             0.10,
            "usefulness":            0.04,
            "constraint_compliance": 0.02,
        },
        min_findings_expected=3,
    ),

    "design": DepartmentContextProfile(
        role="design",
        memory_categories_include=frozenset({
            "goal", "constraint", "research_finding",
            "design_rationale", "rejected_option", "insight", "risk",
        }),
        focus_instruction=(
            "Every design decision needs a stated rationale. "
            "Rejected alternatives must appear in your output with the reason for rejection. "
            "Constraint compliance must be verifiable — name the constraint, show compliance."
        ),
        validation_weights={
            # Completeness (covering tradeoffs) and coherence (logical consistency)
            "structural":            0.18,
            "coherence":             0.20,  # ↑ design logic must hold together
            "completeness":          0.22,  # ↑ must cover all tradeoffs
            "accuracy":              0.18,
            "risk_profile":          0.08,
            "readiness":             0.08,
            "usefulness":            0.04,
            "constraint_compliance": 0.02,
        },
        min_findings_expected=2,
    ),

    "implementation": DepartmentContextProfile(
        role="implementation",
        # Implementation sees approved design and contracts — NOT raw research debates
        memory_categories_include=frozenset({
            "goal", "constraint", "design_rationale", "artifact_reference",
        }),
        focus_instruction=(
            "Work strictly from the approved design specification. "
            "Do not redesign — implement. Every non-obvious decision needs an inline explanation. "
            "Edge cases in the spec must be handled, not skipped."
        ),
        validation_weights={
            # Structural validity and accuracy dominate — did it actually work?
            "structural":            0.25,  # ↑ schema + required fields matter most
            "coherence":             0.12,
            "completeness":          0.20,
            "accuracy":              0.22,  # ↑ does the implementation satisfy the spec?
            "risk_profile":          0.06,
            "readiness":             0.08,
            "usefulness":            0.04,
            "constraint_compliance": 0.03,
        },
        min_findings_expected=2,
    ),

    "qa": DepartmentContextProfile(
        role="qa",
        memory_categories_include=frozenset({
            "goal", "constraint", "test_result",
            "failure_reason", "artifact_reference",
        }),
        focus_instruction=(
            "Approach every artifact with the assumption that it contains flaws. "
            "Cover: happy path, boundary conditions, failure modes, and untested assumptions. "
            "Every defect must be specific — what failed, where, why, severity. "
            "False negatives (missed defects) are worse than false positives."
        ),
        validation_weights={
            # Accuracy is paramount — did QA actually find real problems?
            "structural":            0.20,
            "coherence":             0.12,
            "completeness":          0.20,
            "accuracy":              0.28,  # ↑ highest — QA accuracy IS the product
            "risk_profile":          0.08,
            "readiness":             0.08,
            "usefulness":            0.02,
            "constraint_compliance": 0.02,
        },
        min_findings_expected=3,  # QA must identify multiple items
    ),

    "red_team": DepartmentContextProfile(
        role="red_team",
        memory_categories_include=frozenset(),  # empty = see all — red team needs everything
        focus_instruction=(
            "Your job is to break it. Find attacks the other agents missed. "
            "Every attack vector must be specific and mechanistic: here is how this fails and why. "
            "Distinguish catastrophic failures from recoverable ones. "
            "If you find no serious failures, say so and explain exactly what you tested."
        ),
        validation_weights={
            # Risk profile quality is the entire point of red team
            "structural":            0.15,
            "coherence":             0.12,
            "completeness":          0.18,
            "accuracy":              0.22,
            "risk_profile":          0.20,  # ↑ highest — attack quality is the deliverable
            "readiness":             0.08,
            "usefulness":            0.03,
            "constraint_compliance": 0.02,
        },
        min_findings_expected=3,
    ),

    "integration": DepartmentContextProfile(
        role="integration",
        memory_categories_include=frozenset({
            "goal", "constraint", "research_finding",
            "design_rationale", "artifact_reference", "milestone",
        }),
        focus_instruction=(
            "Resolve every contradiction explicitly — do not silently discard one side. "
            "Preserve well-evidenced minority views. "
            "Document every significant editorial decision you make during the merge."
        ),
        validation_weights={
            # Coherence and completeness: does the merged output hold together and cover all inputs?
            "structural":            0.18,
            "coherence":             0.22,  # ↑ the merge must be internally consistent
            "completeness":          0.22,  # ↑ must represent all upstream inputs
            "accuracy":              0.18,
            "risk_profile":          0.06,
            "readiness":             0.08,
            "usefulness":            0.04,
            "constraint_compliance": 0.02,
        },
        min_findings_expected=2,
    ),

    "documentation": DepartmentContextProfile(
        role="documentation",
        memory_categories_include=frozenset({
            "goal", "constraint", "design_rationale",
            "artifact_reference", "milestone", "decision",
        }),
        focus_instruction=(
            "Write for future retrieval, not the present context. "
            "Every decision must include its rationale, not just its conclusion. "
            "Assume the reader has no prior context. Use precise terminology."
        ),
        validation_weights={
            # Coherence and usefulness matter most — will someone actually use this?
            "structural":            0.18,
            "coherence":             0.22,  # ↑ documentation must be clear and consistent
            "completeness":          0.20,
            "accuracy":              0.15,
            "risk_profile":          0.04,
            "readiness":             0.08,
            "usefulness":            0.10,  # ↑ documentation that doesn't help is worthless
            "constraint_compliance": 0.03,
        },
        min_findings_expected=2,
    ),

    "governance": DepartmentContextProfile(
        role="governance",
        memory_categories_include=frozenset({
            "goal", "constraint", "insight",
            "risk", "milestone", "decision",
        }),
        focus_instruction=(
            "You are conservative by design. "
            "Approve/reject decisions must be explicit and unambiguous. "
            "Every rejection must cite the specific rule or constraint violated. "
            "Uncertainty defaults to 'blocked pending clarification'."
        ),
        validation_weights={
            # Structural clarity and constraint compliance are non-negotiable for governance
            "structural":            0.25,  # ↑ governance output must be unambiguous
            "coherence":             0.15,
            "completeness":          0.18,
            "accuracy":              0.20,
            "risk_profile":          0.08,
            "readiness":             0.06,
            "usefulness":            0.02,
            "constraint_compliance": 0.06,  # ↑ governance must check constraints
        },
        min_findings_expected=2,
    ),

    "orchestration": DepartmentContextProfile(
        role="orchestration",
        memory_categories_include=frozenset({
            "goal", "constraint", "risk",
            "milestone", "decision",
        }),
        focus_instruction=(
            "Decompositions must be exhaustive — no important work left implicit. "
            "Every task needs clear inputs, outputs, and validation criteria. "
            "Dependencies must be explicit — no hidden ordering assumptions."
        ),
        validation_weights={
            # Completeness and structural validity: every task must be accounted for
            "structural":            0.22,
            "coherence":             0.15,
            "completeness":          0.25,  # ↑ plans must cover all required work
            "accuracy":              0.18,
            "risk_profile":          0.08,
            "readiness":             0.08,
            "usefulness":            0.02,
            "constraint_compliance": 0.02,
        },
        min_findings_expected=2,
    ),
}

# Fallback profile used when a department string is unrecognised
_DEFAULT_PROFILE = DepartmentContextProfile(
    role="unknown",
    memory_categories_include=frozenset(),  # see all
    focus_instruction="",
    validation_weights=_DEFAULT_WEIGHTS.copy(),
    min_findings_expected=2,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_department_profile(role: str) -> DepartmentContextProfile:
    """
    Return the profile for the given DepartmentOwner value.
    Never raises — returns _DEFAULT_PROFILE for unknown roles.
    """
    return DEPARTMENT_PROFILES.get(role.lower(), _DEFAULT_PROFILE)


def effective_weights(profile: DepartmentContextProfile) -> dict[str, float]:
    """
    Return the normalised 8-dimension weight dict for a profile.

    Profiles define full weight dicts that should already sum to 1.0, but
    we normalise anyway to guard against authoring errors.
    """
    raw = profile.validation_weights or _DEFAULT_WEIGHTS
    total = sum(raw.values())
    if total == 0:
        return _DEFAULT_WEIGHTS.copy()
    return {k: round(v / total, 6) for k, v in raw.items()}
