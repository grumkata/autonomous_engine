"""
orchestrator/tools/registry.py — Tool definitions, access control, and dispatcher.

MANDATORY PRE-RESPONSE CAPABILITY CHECK
----------------------------------------
Before producing any final response, every agent MUST:
  1. Assess what information they actually need vs. what they have
  2. Check if any available tool can close that gap
  3. Call those tools FIRST, then produce the response

This is enforced by injecting a capability check preamble into every
system prompt, and by the agent_runner tool loop which executes tool_calls
before accepting the response as final.

Tool catalogue:
  WEB:       web_search, fetch_url
  SHELL:     shell (persistent Linux bash workspace)
  CODE:      execute_code (single-script Python)
  VISION:    read_image, read_pdf, read_spreadsheet, screenshot_url
  GENERATE:  generate_image, create_diagram, create_chart
  FILES:     create_spreadsheet, create_document, read_file
  AUDIO:     text_to_speech
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class ToolParam:
    name:        str
    type:        str
    description: str
    required:    bool = True
    default:     Any  = None


@dataclass
class ToolDefinition:
    name:        str
    description: str
    params:      list[ToolParam]
    allowed_departments: set[str] = field(default_factory=lambda: {"*"})

    def to_prompt_block(self) -> str:
        params_text = "\n".join(
            f"    - {p.name} ({p.type}{'?' if not p.required else ''}): {p.description}"
            for p in self.params
        )
        return (
            f"**{self.name}**\n"
            f"  {self.description}\n"
            f"  Parameters:\n{params_text}"
        )


# ---------------------------------------------------------------------------
# Capability check — injected into every system prompt
# ---------------------------------------------------------------------------

CAPABILITY_CHECK_PREAMBLE = """
## MANDATORY CAPABILITY CHECK — Do this BEFORE producing any response

Before you write your final JSON response, you MUST complete this internal check:

1. **What do I need to answer this well?**
   - Facts, data, or current information I don't have? → use `web_search` or `fetch_url`
   - Files I need to read or inspect? → use `read_file`, `read_pdf`, or `read_spreadsheet`
   - Images or visuals I need to see? → use `read_image` or `screenshot_url`
   - Code I need to run or test? → use `shell` or `execute_code`
   - Diagrams, charts, or files I need to produce? → use the appropriate generation tool

2. **Am I about to state something I should verify?**
   If yes → call `web_search` first.

3. **Would seeing the actual file/image/page improve my answer?**
   If yes → call the appropriate vision tool first.

4. **Only produce your final JSON AFTER you have gathered what you need.**

To use tools, include a `tool_calls` array in your response:
```json
{
  "tool_calls": [
    {"tool": "web_search", "params": {"query": "your query here"}},
    {"tool": "shell",      "params": {"command": "ls -la && python3 --version"}}
  ]
}
```

Tool results will be fed back to you automatically. You will then produce your final response.
You may chain multiple rounds of tool calls if needed.
""".strip()


# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------

_ALL_TOOLS: list[ToolDefinition] = [

    # ── Web ──────────────────────────────────────────────────────────────────

    ToolDefinition(
        name="web_search",
        description="Search the web for current information, docs, research, or examples. Use BEFORE stating any fact you're not certain of.",
        params=[
            ToolParam("query",       "string", "The search query"),
            ToolParam("num_results", "number", "Results to return (1-10, default 5)", required=False, default=5),
        ],
        allowed_departments={"*"},
    ),

    ToolDefinition(
        name="fetch_url",
        description="Fetch and read the full content of a webpage. Use after web_search to read specific pages, docs, or articles in full.",
        params=[
            ToolParam("url",       "string", "Full URL starting with http:// or https://"),
            ToolParam("max_chars", "number", "Max characters to return (default 8000)", required=False, default=8000),
        ],
        allowed_departments={"*"},
    ),

    # ── Shell workspace ───────────────────────────────────────────────────────

    ToolDefinition(
        name="shell",
        description=(
            "Run any bash command in a persistent Linux workspace. The shell maintains state "
            "between calls (environment, working directory, installed packages). "
            "Use for: running code, installing packages, creating/processing files, "
            "git operations, curl, data processing, anything a developer would do in a terminal. "
            "Files created here are automatically tracked in the project workspace."
        ),
        params=[
            ToolParam("command",         "string", "Bash command(s) to execute. Can be multi-line."),
            ToolParam("timeout_seconds", "number", "Timeout in seconds (default 30, max 300)", required=False, default=30),
        ],
        allowed_departments={"implementation", "qa", "red_team", "integration", "research", "design"},
    ),

    # ── Code execution ────────────────────────────────────────────────────────

    ToolDefinition(
        name="execute_code",
        description="Run a self-contained Python script. For interactive or stateful work, use `shell` instead.",
        params=[
            ToolParam("code",            "string", "Python code to execute"),
            ToolParam("timeout_seconds", "number", "Max execution time (default 30)", required=False, default=30),
        ],
        allowed_departments={"implementation", "qa", "red_team", "integration", "research"},
    ),

    # ── Vision — agents can SEE files and pages ───────────────────────────────

    ToolDefinition(
        name="read_image",
        description=(
            "Load an image from the project workspace so you can visually inspect it. "
            "Use to examine generated images, diagrams, screenshots, or any visual artifact."
        ),
        params=[
            ToolParam("path",      "string", "File path relative to project workspace root"),
            ToolParam("max_width", "number", "Max width in pixels (default 1024)", required=False, default=1024),
        ],
        allowed_departments={"*"},
    ),

    ToolDefinition(
        name="read_pdf",
        description=(
            "Render PDF pages as images so you can read them visually. "
            "Use to inspect PDF documents, reports, research papers, or any PDF in the workspace."
        ),
        params=[
            ToolParam("path",  "string", "PDF file path relative to workspace root"),
            ToolParam("pages", "string", "Pages to render: '1', '1-3', 'all' (default '1-3')", required=False, default="1-3"),
        ],
        allowed_departments={"*"},
    ),

    ToolDefinition(
        name="read_spreadsheet",
        description=(
            "Render an Excel spreadsheet as a visual table so you can inspect its contents. "
            "Also returns raw data. Use to examine data files, reports, or structured data."
        ),
        params=[
            ToolParam("path",     "string", "Excel file path relative to workspace root"),
            ToolParam("sheet",    "string", "Sheet name (default: first sheet)", required=False, default=None),
            ToolParam("max_rows", "number", "Max rows to show (default 50)", required=False, default=50),
        ],
        allowed_departments={"*"},
    ),

    ToolDefinition(
        name="screenshot_url",
        description=(
            "Take a screenshot of any webpage for visual inspection. "
            "Use to see what a website looks like, inspect UI, or verify web content visually."
        ),
        params=[
            ToolParam("url",     "string", "URL to screenshot"),
            ToolParam("wait_ms", "number", "Wait time after page load in ms (default 1500)", required=False, default=1500),
        ],
        allowed_departments={"design", "qa", "research", "red_team", "implementation"},
    ),

    # ── Image / diagram generation ────────────────────────────────────────────

    ToolDefinition(
        name="generate_image",
        description="Generate an image from a text prompt using AI. Saves to project workspace.",
        params=[
            ToolParam("prompt",  "string", "Detailed image description"),
            ToolParam("style",   "string", "'realistic', 'illustration', 'diagram', 'logo', 'sketch'", required=False, default="realistic"),
            ToolParam("width",   "number", "Width in pixels (256-1024)", required=False, default=512),
            ToolParam("height",  "number", "Height in pixels (256-1024)", required=False, default=512),
        ],
        allowed_departments={"design", "documentation", "implementation"},
    ),

    ToolDefinition(
        name="create_diagram",
        description="Create a Mermaid or Graphviz diagram and render it to PNG.",
        params=[
            ToolParam("definition", "string", "Mermaid or Graphviz DOT diagram definition"),
            ToolParam("format",     "string", "'mermaid' or 'graphviz'", required=False, default="mermaid"),
            ToolParam("filename",   "string", "Output filename without extension", required=False, default="diagram"),
        ],
        allowed_departments={"design", "implementation", "documentation", "research"},
    ),

    ToolDefinition(
        name="create_chart",
        description="Create a data chart (bar, line, pie, scatter, histogram, heatmap) and save as PNG.",
        params=[
            ToolParam("chart_type", "string", "'bar', 'line', 'pie', 'scatter', 'histogram', 'heatmap'"),
            ToolParam("data",       "object", "Data: {labels: [...], datasets: [{label, values: [...]}]}"),
            ToolParam("title",      "string", "Chart title"),
            ToolParam("filename",   "string", "Output filename without extension", required=False, default="chart"),
        ],
        allowed_departments={"research", "design", "documentation", "qa", "implementation"},
    ),

    # ── File operations ───────────────────────────────────────────────────────

    ToolDefinition(
        name="create_spreadsheet",
        description="Create an Excel spreadsheet (.xlsx) with formatted sheets of data.",
        params=[
            ToolParam("sheets",   "array",  "List of sheets: [{name, headers: [...], rows: [[...]]}]"),
            ToolParam("filename", "string", "Output filename without .xlsx extension"),
        ],
        allowed_departments={"research", "documentation", "implementation", "qa"},
    ),

    ToolDefinition(
        name="create_document",
        description="Create a formatted text document (.md or .txt) and save to workspace.",
        params=[
            ToolParam("content",  "string", "Full document content (Markdown supported)"),
            ToolParam("filename", "string", "Output filename without extension"),
            ToolParam("format",   "string", "'md' or 'txt'", required=False, default="md"),
        ],
        allowed_departments={"*"},
    ),

    ToolDefinition(
        name="read_file",
        description="Read a text file from the project workspace. Use read_pdf or read_spreadsheet for those types.",
        params=[
            ToolParam("path",      "string", "File path relative to project workspace root"),
            ToolParam("max_chars", "number", "Max characters to return (default 10000)", required=False, default=10000),
        ],
        allowed_departments={"*"},
    ),

    # ── Audio ─────────────────────────────────────────────────────────────────

    ToolDefinition(
        name="text_to_speech",
        description="Convert text to speech and save as MP3 in the project workspace.",
        params=[
            ToolParam("text",     "string", "Text to convert to speech"),
            ToolParam("voice",    "string", "'neutral', 'male', 'female', 'british', 'american'", required=False, default="neutral"),
            ToolParam("filename", "string", "Output filename without .mp3 extension", required=False, default="audio"),
        ],
        allowed_departments={"design", "documentation", "implementation"},
    ),
]

_TOOL_MAP: dict[str, ToolDefinition] = {t.name: t for t in _ALL_TOOLS}


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

def get_tools_for_department(department: str) -> list[ToolDefinition]:
    return [
        t for t in _ALL_TOOLS
        if "*" in t.allowed_departments or department in t.allowed_departments
    ]


def get_tool(name: str) -> ToolDefinition | None:
    return _TOOL_MAP.get(name)


def build_tools_prompt(department: str) -> str:
    """Build the tools + capability check section for the system prompt."""
    tools = get_tools_for_department(department)
    if not tools:
        return ""

    lines = [CAPABILITY_CHECK_PREAMBLE, "", "---", "", "## Your Available Tools\n"]
    for tool in tools:
        lines.append(tool.to_prompt_block())
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def execute_tool(
    tool_name:  str,
    params:     dict[str, Any],
    project_id: str,
    task_id:    str,
    department: str,
) -> dict[str, Any]:
    tool_def = get_tool(tool_name)
    if not tool_def:
        return {"tool": tool_name, "success": False,
                "error": f"Unknown tool: '{tool_name}'. Available: {list(_TOOL_MAP.keys())}",
                "files_created": [], "visual_blocks": []}

    if "*" not in tool_def.allowed_departments and department not in tool_def.allowed_departments:
        return {"tool": tool_name, "success": False,
                "error": f"Department '{department}' does not have access to '{tool_name}'",
                "files_created": [], "visual_blocks": []}

    log.info("tool.execute", tool=tool_name, project_id=project_id,
             task_id=task_id, department=department)

    try:
        # ── Web ──────────────────────────────────────────────────────────
        if tool_name == "web_search":
            from orchestrator.tools.web import web_search
            return await web_search(project_id=project_id, **params)

        elif tool_name == "fetch_url":
            from orchestrator.tools.web import fetch_url
            return await fetch_url(project_id=project_id, **params)

        # ── Shell workspace ───────────────────────────────────────────────
        elif tool_name == "shell":
            from orchestrator.tools.workspace_shell import shell_run
            return await shell_run(project_id=project_id, task_id=task_id, **params)

        # ── Code execution ────────────────────────────────────────────────
        elif tool_name == "execute_code":
            from orchestrator.tools.code_exec import execute_code
            return await execute_code(project_id=project_id, task_id=task_id, **params)

        # ── Vision ────────────────────────────────────────────────────────
        elif tool_name == "read_image":
            from orchestrator.tools.vision import read_image
            return await read_image(project_id=project_id, **params)

        elif tool_name == "read_pdf":
            from orchestrator.tools.vision import read_pdf
            return await read_pdf(project_id=project_id, **params)

        elif tool_name == "read_spreadsheet":
            from orchestrator.tools.vision import read_spreadsheet
            return await read_spreadsheet(project_id=project_id, **params)

        elif tool_name == "screenshot_url":
            from orchestrator.tools.vision import screenshot_url
            return await screenshot_url(project_id=project_id, **params)

        # ── Image / diagram generation ────────────────────────────────────
        elif tool_name == "generate_image":
            from orchestrator.tools.image import generate_image
            return await generate_image(project_id=project_id, **params)

        elif tool_name == "create_diagram":
            from orchestrator.tools.image import create_diagram
            return await create_diagram(project_id=project_id, **params)

        elif tool_name == "create_chart":
            from orchestrator.tools.image import create_chart
            return await create_chart(project_id=project_id, **params)

        # ── File operations ───────────────────────────────────────────────
        elif tool_name == "create_spreadsheet":
            from orchestrator.tools.files import create_spreadsheet
            return await create_spreadsheet(project_id=project_id, **params)

        elif tool_name == "create_document":
            from orchestrator.tools.files import create_document
            return await create_document(project_id=project_id, **params)

        elif tool_name == "read_file":
            from orchestrator.tools.files import read_file
            return await read_file(project_id=project_id, **params)

        # ── Audio ─────────────────────────────────────────────────────────
        elif tool_name == "text_to_speech":
            from orchestrator.tools.audio import text_to_speech
            return await text_to_speech(project_id=project_id, **params)

        else:
            return {"tool": tool_name, "success": False,
                    "error": f"Tool '{tool_name}' dispatcher not implemented.",
                    "files_created": [], "visual_blocks": []}

    except Exception as exc:
        log.error("tool.error", tool=tool_name, project_id=project_id, error=str(exc))
        return {"tool": tool_name, "success": False,
                "error": f"Tool execution failed: {exc}",
                "files_created": [], "visual_blocks": []}
