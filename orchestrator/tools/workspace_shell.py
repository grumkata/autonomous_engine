"""
orchestrator/tools/workspace_shell.py — Persistent Linux bash shell per project.

Every project gets a long-lived bash subprocess that maintains state between
tool calls within a task (environment variables, working directory, installed
packages, etc.).  This is analogous to how Claude computer-use works — the AI
has a real shell it can use persistently across multiple commands.

Key features:
  - Persistent between tool calls within the same task session
  - Stateful: cd, export, pip install all persist
  - Working directory: ./workspace_files/{project_id}/shell/
  - Pre-loaded with useful tools: python3, pip, curl, jq, git, etc.
  - Hard timeout per command (default 30s)
  - Output captured and returned; files created in workspace are tracked
  - Shell dies when the task completes or on timeout; auto-restarts next call

Shell lifecycle:
  get_shell(project_id) → ProjectShell (singleton per project)
  shell.run("command") → ShellResult
  shell.close()        → terminate shell process

Security notes:
  - Runs as the engine process user (same user as the engine)
  - Network access: SAME as the engine (not restricted by default)
  - File system: full access to workspace_files/{project_id}/shell/
  - For stronger isolation: set SHELL_USE_DOCKER=true (planned)
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from orchestrator.tools.file_store import project_dir, save_bytes, ResourceType, _EXT_TO_TYPE

log = structlog.get_logger(__name__)

_DEFAULT_TIMEOUT  = 30      # seconds per command
_MAX_TIMEOUT      = 300     # hard cap
_MAX_OUTPUT_CHARS = 40_000  # truncate if shell floods

# Global registry: project_id → ProjectShell
_shells: dict[str, "ProjectShell"] = {}


# ---------------------------------------------------------------------------
# Shell result
# ---------------------------------------------------------------------------

@dataclass
class ShellResult:
    command:       str
    stdout:        str
    stderr:        str
    exit_code:     int
    timed_out:     bool = False
    files_created: list[str] = field(default_factory=list)
    cwd:           str = ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_tool_result(self) -> dict[str, Any]:
        output = self.stdout
        if self.stderr:
            output += f"\n[stderr]\n{self.stderr}"
        if self.timed_out:
            output += f"\n[TIMED OUT after command exceeded limit]"
        return {
            "tool":          "shell",
            "success":       self.success,
            "exit_code":     self.exit_code,
            "output":        output[:_MAX_OUTPUT_CHARS],
            "cwd":           self.cwd,
            "files_created": self.files_created,
            "error":         f"Exit code {self.exit_code}" if self.exit_code != 0 else "",
        }


# ---------------------------------------------------------------------------
# ProjectShell — one persistent bash process per project
# ---------------------------------------------------------------------------

class ProjectShell:
    """A persistent bash subprocess for a project workspace."""

    def __init__(self, project_id: str) -> None:
        self.project_id  = project_id
        self._proc:      asyncio.subprocess.Process | None = None
        self._lock:      asyncio.Lock = asyncio.Lock()
        self._shell_dir: Path = project_dir(project_id) / "shell"
        self._shell_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot:  set[str] = set()

    async def _ensure_running(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.returncode is None:
                    return  # still alive
            except Exception:
                pass

        # Start a fresh bash process
        env = os.environ.copy()
        env["PS1"] = ""               # no prompt noise
        env["TERM"] = "dumb"
        env["PYTHONUNBUFFERED"] = "1"
        # Point pip installs to local dir so they don't need sudo
        local_pip = str(self._shell_dir / ".local")
        env["PYTHONUSERBASE"] = local_pip
        env["PATH"] = f"{local_pip}/bin:{env.get('PATH', '/usr/bin:/bin')}"

        self._proc = await asyncio.create_subprocess_exec(
            "bash", "--norc", "--noprofile",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._shell_dir),
            env=env,
        )

        # Set up the shell environment
        init_cmds = [
            f"cd {self._shell_dir}",
            "export PYTHONDONTWRITEBYTECODE=1",
            "alias python=python3",
            f"echo SHELL_READY",
        ]
        for cmd in init_cmds[:-1]:
            self._proc.stdin.write(f"{cmd}\n".encode())
        self._proc.stdin.write(f"{init_cmds[-1]}\n".encode())
        await self._proc.stdin.drain()

        # Wait for SHELL_READY
        try:
            while True:
                line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=10.0)
                if b"SHELL_READY" in line:
                    break
        except asyncio.TimeoutError:
            log.warning("workspace_shell.init_timeout", project_id=self.project_id)

        self._snapshot = self._file_snapshot()
        log.info("workspace_shell.started", project_id=self.project_id)

    async def run(
        self,
        command: str,
        timeout_seconds: int = _DEFAULT_TIMEOUT,
    ) -> ShellResult:
        """Run a shell command and return the result."""
        timeout_seconds = max(5, min(timeout_seconds, _MAX_TIMEOUT))
        sentinel = f"__END_{uuid.uuid4().hex[:8]}__"

        async with self._lock:
            await self._ensure_running()
            assert self._proc is not None

            before = self._file_snapshot()

            # Write command + exit-code capture + sentinel
            # NOTE: do NOT wrap in () — that creates a subshell and kills
            # environment variable persistence between calls
            full_cmd = (
                f"{command}\n"
                f"_exit_code=$?\n"
                f"echo {sentinel}_EXIT_$_exit_code\n"
            )
            self._proc.stdin.write(full_cmd.encode())
            await self._proc.stdin.drain()

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            exit_code    = 0
            timed_out    = False

            # Drain stdout until sentinel
            try:
                while True:
                    line = await asyncio.wait_for(
                        self._proc.stdout.readline(), timeout=float(timeout_seconds)
                    )
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    if sentinel in decoded:
                        # Extract exit code
                        m = re.search(r"_EXIT_(\d+)", decoded)
                        if m:
                            exit_code = int(m.group(1))
                        break
                    stdout_lines.append(decoded)
            except asyncio.TimeoutError:
                timed_out = True
                self._proc.kill()
                self._proc = None   # will restart on next call

            # Drain any pending stderr (non-blocking)
            try:
                raw_err = await asyncio.wait_for(
                    self._drain_stderr(), timeout=0.5
                )
                if raw_err:
                    stderr_lines = raw_err.splitlines()
            except asyncio.TimeoutError:
                pass

            # Get CWD for context
            cwd = str(self._shell_dir)
            if not timed_out and self._proc and self._proc.returncode is None:
                try:
                    cwd = await self._get_cwd()
                except Exception:
                    pass

            # Detect new files
            after    = self._file_snapshot()
            new_keys = after - before
            new_files = self._register_new_files(new_keys)

        stdout = "\n".join(stdout_lines)[:_MAX_OUTPUT_CHARS]
        stderr = "\n".join(stderr_lines)[:5000]

        log.info(
            "workspace_shell.command",
            project_id=self.project_id,
            exit_code=exit_code,
            timed_out=timed_out,
            new_files=len(new_files),
            cmd_preview=command[:80],
        )

        return ShellResult(
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            files_created=new_files,
            cwd=cwd,
        )

    async def _drain_stderr(self) -> str:
        """Read all available stderr without blocking indefinitely."""
        if not self._proc or not self._proc.stderr:
            return ""
        chunks = []
        try:
            while True:
                chunk = await asyncio.wait_for(self._proc.stderr.read(4096), timeout=0.1)
                if not chunk:
                    break
                chunks.append(chunk.decode("utf-8", errors="replace"))
        except (asyncio.TimeoutError, Exception):
            pass
        return "".join(chunks)

    async def _get_cwd(self) -> str:
        """Query current working directory of the shell."""
        sentinel = f"__CWD_{uuid.uuid4().hex[:6]}__"
        self._proc.stdin.write(f"echo {sentinel}:$(pwd)\n".encode())
        await self._proc.stdin.drain()
        try:
            while True:
                line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=2.0)
                decoded = line.decode().strip()
                if sentinel in decoded:
                    return decoded.split(":", 1)[1] if ":" in decoded else str(self._shell_dir)
        except asyncio.TimeoutError:
            return str(self._shell_dir)

    def _file_snapshot(self) -> set[str]:
        """Snapshot all files under the shell directory."""
        try:
            return {
                str(p.relative_to(self._shell_dir))
                for p in self._shell_dir.rglob("*")
                if p.is_file() and ".local" not in str(p)
            }
        except Exception:
            return set()

    def _register_new_files(self, relative_keys: set[str]) -> list[str]:
        """Copy new files to appropriate workspace type dirs."""
        registered = []
        for rel in relative_keys:
            src = self._shell_dir / rel
            if not src.is_file():
                continue
            try:
                data = src.read_bytes()
                ext  = src.suffix.lower()
                rtype = _EXT_TO_TYPE.get(ext, ResourceType.RAW)
                dest = save_bytes(self.project_id, src.name, data, rtype)
                dest_rel = str(dest.relative_to(project_dir(self.project_id)))
                registered.append(dest_rel)
            except Exception as exc:
                log.warning("workspace_shell.file_register_failed", file=rel, error=str(exc))
                registered.append(f"shell/{rel}")
        return registered

    async def close(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.kill()
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except Exception:
                pass
        self._proc = None
        log.info("workspace_shell.closed", project_id=self.project_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_shell(project_id: str) -> ProjectShell:
    """Get or create the persistent shell for a project."""
    if project_id not in _shells:
        _shells[project_id] = ProjectShell(project_id)
    return _shells[project_id]


async def close_shell(project_id: str) -> None:
    """Close and remove the shell for a project."""
    if project_id in _shells:
        await _shells[project_id].close()
        del _shells[project_id]


async def close_all_shells() -> None:
    """Close all active shells — call on engine shutdown."""
    for shell in list(_shells.values()):
        await shell.close()
    _shells.clear()


# ---------------------------------------------------------------------------
# Tool handler (called from registry.execute_tool)
# ---------------------------------------------------------------------------


async def shell_run(
    command: str,
    project_id: str,
    task_id: str,
    timeout_seconds: int = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Tool entry point: run a shell command in the project workspace."""
    shell = get_shell(project_id)
    result = await shell.run(command, timeout_seconds=timeout_seconds)
    return result.to_tool_result()
