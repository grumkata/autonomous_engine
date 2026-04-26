# SKILL: qa/security — Security Review & Red Team

## Purpose
Guides red team agents performing security reviews, threat modelling,
and adversarial analysis of systems, designs, or code.

## Attack surface taxonomy — check all categories
1. **Input validation** — injection (SQL, command, path traversal), parsing bugs
2. **Authentication & authorisation** — missing auth, privilege escalation, IDOR
3. **Data exposure** — secrets in logs/code, over-returning data, PII leakage
4. **Dependency risk** — known CVEs, outdated packages, supply chain
5. **Logic flaws** — race conditions, TOCTOU, business logic bypasses
6. **Denial of service** — unbounded loops, resource exhaustion, large payloads
7. **Cryptography** — weak algorithms, key management, IV reuse
8. **Configuration** — debug mode in prod, permissive CORS, open ports

## Finding format
For every vulnerability found:
- **VULN-ID**: sequential (VULN-001...)
- **Category**: from taxonomy above
- **Severity**: Critical / High / Medium / Low / Informational
- **Attack vector**: how an attacker triggers this
- **Impact**: what they can achieve if successful
- **Reproduction**: minimal steps to demonstrate
- **Remediation**: specific fix, not "sanitize inputs"

## What a clean review looks like
If you find no serious vulnerabilities, explicitly state:
- Each category you checked
- What evidence/code you inspected for each
- Why you concluded it was safe

"No issues found" without that evidence is not a clean review.

## What to include in findings
- Total vulnerabilities by severity.
- Highest-severity finding summarised.
- Attack categories with no findings (proof of coverage).
- Most impactful remediation (if you could fix only one thing).

## Common failure modes — avoid these
- Generic findings: "SQL injection is possible" without showing where.
- Missing the non-obvious attack vectors (focus on OWASP Top 10 is not enough).
- Treating low severity as not worth reporting — attackers chain them.
- Recommending "add authentication" without specifying what mechanism.
- Skipping the authorisation review because authentication exists.
