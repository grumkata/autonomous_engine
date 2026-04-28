"""
orchestrator/tools/code_exec.py — Sandboxed Python code execution.

Runs LLM-generated Python in a subprocess with:
  - Hard timeout (default 30s, max 120s)
  - Working directory set to the project workspace
  - stdout/stderr captured
  - Output files collected and registered in the file store
  - No network access from within the subprocess (OS-level restriction via env)

Safety notes:
  - Subprocess isolation: the code runs in a separate process.
    If it crashes, the engine continues.
  - No internet: REQUESTS_CA_BUNDLE="" prevents most HTTP calls.
    This is soft — not a hard sandbox like Docker.
  - File system: code runs in ./workspace_files/{project_id}/code_exec/
    so it can write files there but can't easily escape.

For stronger isolation, set CODE_EXEC_USE_DOCKER=true in .env.
Docker support is planned but not yet implemented — contributions welcome.
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import structlog

from orchestrator.tools.file_store import project_dir, list_files, ResourceType

log = structlog.get_logger(__name__)

_MAX_TIMEOUT = 120
_MAX_OUTPUT_CHARS = 20_000


async def execute_code(
    code: str,
    project_id: str,
    task_id: str,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """
    Execute Python code and return stdout, stderr, and any files created.
    """
    timeout_seconds = max(5, min(timeout_seconds, _MAX_TIMEOUT))

    # Prepare sandbox directory
    exec_dir = project_dir(project_id) / "code_exec"
    exec_dir.mkdir(parents=True, exist_ok=True)

    # Write the code to a temp file
    script_path = exec_dir / f"exec_{task_id[:8]}.py"
    script_path.write_text(code, encoding="utf-8")

    # Snapshot existing files before execution
    files_before = _snapshot_files(exec_dir)

    # Environment: restrict network, point to project dir
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = ""
    # Soft network block — prevents accidental external calls
    env["REQUESTS_CA_BUNDLE"] = ""
    env["CURL_CA_BUNDLE"] = ""

    log.info("code_exec.start", project_id=project_id, task_id=task_id,
             timeout=timeout_seconds, lines=code.count("\n") + 1)

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(exec_dir),
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout_seconds)
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "tool": "execute_code",
                "success": False,
                "error": f"Code execution timed out after {timeout_seconds}s",
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "files_created": [],
            }

        stdout = stdout_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
        exit_code = proc.returncode

    except Exception as exc:
        return {
            "tool": "execute_code",
            "success": False,
            "error": f"Failed to launch subprocess: {exc}",
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "files_created": [],
        }
    finally:
        # Clean up the script file
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Detect new files created during execution
    files_after = _snapshot_files(exec_dir)
    new_files = [f for f in files_after if f not in files_before]

    # Move any new files to appropriate workspace dirs and collect paths
    registered_files: list[str] = []
    for new_file in new_files:
        src = exec_dir / new_file
        if src.exists():
            # Copy to appropriate type directory in workspace
            try:
                from orchestrator.tools.file_store import save_bytes, _EXT_TO_TYPE, ResourceType
                data = src.read_bytes()
                ext = src.suffix.lower()
                rtype = _EXT_TO_TYPE.get(ext, ResourceType.RAW)
                dest = save_bytes(project_id, new_file, data, rtype)
                registered_files.append(str(dest.relative_to(project_dir(project_id))))
                src.unlink(missing_ok=True)
            except Exception as exc:
                log.warning("code_exec.file_copy_failed", file=new_file, error=str(exc))
                registered_files.append(f"code_exec/{new_file}")

    log.info(
        "code_exec.complete",
        project_id=project_id,
        task_id=task_id,
        exit_code=exit_code,
        new_files=len(new_files),
    )

    success = exit_code == 0

    result = {
        "tool": "execute_code",
        "success": success,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr if stderr else "",
        "files_created": registered_files,
    }
    if not success and stderr:
        result["error"] = f"Code exited with code {exit_code}. See stderr."

    return result


def _snapshot_files(directory: Path) -> set[str]:
    """Return set of filenames currently in directory (non-recursive)."""
    try:
        return {f.name for f in directory.iterdir() if f.is_file()}
    except Exception:
        return set()
