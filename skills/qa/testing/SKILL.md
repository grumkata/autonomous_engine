# SKILL: qa/testing — Test Case Generation & Quality Assurance

## Purpose
Guides QA agents generating test cases, test plans, and defect reports.

## Test coverage mandate — all four quadrants required
1. **Happy path** — inputs in valid range, expected outputs produced
2. **Boundary conditions** — exactly at limits, one above, one below
3. **Failure modes** — invalid inputs, missing required fields, type mismatches
4. **Edge cases** — empty collections, concurrent access, extremely large/small values

A test suite missing any quadrant is incomplete regardless of line count.

## Defect report format
Every defect must contain:
- **ID**: sequential (DEF-001, DEF-002...)
- **Severity**: Critical / High / Medium / Low
- **Location**: specific function, line, or component
- **Steps to reproduce**: numbered, precise, reproducible
- **Expected**: what should happen
- **Actual**: what does happen
- **Root cause hypothesis**: your best analysis of why

Do not report "it might fail" — report observed or logically certain failures.

## Test case format
```
TEST-001: [Description]
  Given: [preconditions]
  When:  [action]
  Then:  [expected outcome]
  Notes: [edge case rationale]
```

## What to include in findings
- Total test cases by quadrant (happy/boundary/failure/edge).
- Critical defects found (DEF-### list).
- Coverage gaps — what you could not test and why.
- Pass/fail verdict with justification.

## Common failure modes — avoid these
- Testing only the happy path and calling it complete.
- Vague defects: "it might not work under load" without specifics.
- Duplicate test cases that test the same thing differently worded.
- Skipping error handling paths because they're "unlikely".
- Reporting opinion as defect ("I don't like this API design").
