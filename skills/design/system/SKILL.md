# SKILL: design/system — System Architecture & Design

## Purpose
Guides design agents producing system architecture documents, component
specifications, API contracts, data schemas, or technical design documents.

## Required output sections
Every system design output must contain:
1. **Problem statement** — what specific problem this design solves
2. **Design decisions** — each decision with explicit rationale
3. **Rejected alternatives** — what you considered and why you didn't choose it
4. **Component breakdown** — named components with clear responsibilities
5. **Interface contracts** — how components communicate (API shapes, event schemas, etc.)
6. **Failure modes** — what happens when each component fails
7. **Open questions** — decisions deferred and why

## Quality bar
- Every component must have exactly one clearly stated responsibility.
- If two components have overlapping responsibilities, flag it as a design smell.
- Interface contracts must be concrete: show actual field names, types, and semantics.
- Rejected alternatives must include a specific reason for rejection — not just "too complex".
- "We'll figure it out later" is not acceptable — name what needs to be decided and by whom.

## Diagram format (text-based)
When depicting component relationships, use ASCII or structured text:
```
[ComponentA] --{event: UserCreated}--> [ComponentB]
[ComponentB] --{HTTP POST /api/things}--> [ComponentC]
```

## What to include in findings
- The list of named components and their single responsibility each.
- The key tradeoff you made and why.
- The constraint that most shaped the design.
- What you explicitly left out of scope.

## Common failure modes — avoid these
- God components (one thing that does everything).
- Interface contracts described in natural language without field definitions.
- Designing for the happy path only — no failure mode analysis.
- Presenting one design option as if alternatives don't exist.
- "Microservices" as a default without justification.
