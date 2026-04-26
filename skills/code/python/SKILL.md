# SKILL: code/python — Python Implementation

## Purpose
Guides implementation agents producing Python code artifacts.
Read this entire document before writing any code.

## Environment constraints
- Python 3.11+. Use `from __future__ import annotations` in every file.
- Type annotations are mandatory on every function signature.
- Async-first: prefer `async def` for I/O-bound work; use `asyncio` not threads.
- Dependencies must already exist in requirements.txt — do not invent new ones.

## Code quality bar
- Every public function/class needs a docstring (one-line minimum).
- No bare `except:` — always catch specific exceptions.
- No `print()` for logging — use `structlog.get_logger(__name__)`.
- Magic numbers must be named constants or explained inline.
- `TODO` comments are only acceptable if they reference a known blocker; explain it.

## Output format
Produce complete, runnable Python files. Structure:
```
"""Module docstring."""
from __future__ import annotations
# stdlib
# third-party
# local
# constants
# classes / functions
```

## What to include in findings
- The module/function names you created and what each does.
- Any design decisions (e.g. why async, why a particular data structure).
- Edge cases you handled and how.
- What you did NOT implement and why (if scope limited you).

## Common failure modes — avoid these
- Returning placeholder stubs without explaining what they need.
- Mixing sync and async incorrectly (calling sync blocking code in an async loop).
- Missing error handling on I/O paths (DB, HTTP, file system).
- Producing code that imports from modules that don't exist in this project.
