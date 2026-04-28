"""
orchestrator/tools/registry.py — Tool definitions and per-department access control.

Every tool is defined once here.  The registry controls:
  - What parameters a tool accepts
  - Which departments have access to which tools
  - The description injected into the agent's system prompt

Tool execution happens in the individual tool modules.
The agent_runner calls execute_tool() which dispatches here.

Department → tool access map:
  research       → web_search, fetch_url, read_file, create_chart
  design         → generate_image, create_diagram, create_chart, web_search
  implementation → execute_code, create_diagram, create_chart, read_file, web_search, fetch_url
  qa             → execute_code, read_file, fetch_url, web_search
  red_team       → web_search, fetch_url, execute_code, read_file
  integration    → execute_code, read_file, fetch_url, web_search
  documentation  → create_pdf, read_file, create_chart, web_search, fetch_url
  governance     → web_search, fetch_url, read_file
  orchestration  → web_search  (planner only needs search)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class ToolParam:
    name: str
    type: str          # "string" | "number" | "boolean" | "object" | "array"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    params: list[ToolParam]
    # Departments that may use this tool ("*" = all)
    allowed_departments: set[str] = field(default_factory=lambda: {"*"})

    def to_prompt_block(self) -> str:
        """Format tool definition for injection into system prompt."""
        params_text = "\n".join(
            f"    - {p.name} ({p.type}{'?' if not p.required else ''}): {p.description}"
            for p in self.params
        )
        return (
            f"Tool: {self.name}\n"
            f"  Description: {self.description}\n"
            f"  Parameters:\n{params_text}"
        )


# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------

_ALL_TOOLS: list[ToolDefinition] = [

    # ── Web ───────────────────────────────────────────────────────────────

    ToolDefinition(
        name="web_search",
        description=(
            "Search the web for current information, news, documentation, "
            "research papers, examples, or any topic. Returns a list of results "
            "with titles, URLs, and snippets. Use this when you need facts, "
            "current data, or external references."
        ),
        params=[
            ToolParam("query", "string", "The search query"),
            ToolParam("num_results", "number", "Number of results to return (1-10)", required=False, default=5),
        ],
        allowed_departments={"*"},
    ),

    ToolDefinition(
        name="fetch_url",
        description=(
            "Fetch and read the full content of a webpage. Returns cleaned text "
            "from the page, stripping ads and navigation. Use after web_search to "
            "read specific pages, documentation, or articles in full."
        ),
        params=[
            ToolParam("url", "string", "The full URL to fetch (must start with http:// or https://)"),
            ToolParam("max_chars", "number", "Max characters to return (default 8000)", required=False, default=8000),
        ],
        allowed_departments={"*"},
    ),

    # ── Code execution ────────────────────────────────────────────────────

    ToolDefinition(
        name="execute_code",
        description=(
            "Execute Python code in a sandboxed environment. Can produce files "
            "as output (charts, data files, processed results). Returns stdout, "
            "stderr, and paths of any files created. Use for: data processing, "
            "calculations, chart generation, file creation, testing logic."
        ),
        params=[
            ToolParam("code", "string", "Python code to execute"),
            ToolParam("timeout_seconds", "number", "Max execution time (default 30, max 120)", required=False, default=30),
        ],
        allowed_departments={
            "implementation", "qa", "red_team", "integration", "research",
        },
    ),

    # ── Image / Visual ────────────────────────────────────────────────────

    ToolDefinition(
        name="generate_image",
        description=(
            "Generate an image from a text prompt using an AI image model. "
            "Returns the filename saved to the project workspace. Use for: "
            "mockups, illustrations, concept art, visual assets."
        ),
        params=[
            ToolParam("prompt", "string", "Detailed description of the image to generate"),
            ToolParam("style", "string", "Art style: 'realistic', 'illustration', 'diagram', 'logo', 'sketch'", required=False, default="realistic"),
            ToolParam("width", "number", "Image width in pixels (256-1024)", required=False, default=512),
            ToolParam("height", "number", "Image height in pixels (256-1024)", required=False, default=512),
        ],
        allowed_departments={"design", "documentation", "implementation"},
    ),

    ToolDefinition(
        name="create_diagram",
        description=(
            "Create a diagram from a Mermaid or Graphviz definition. "
            "Renders to PNG and saves to the project workspace. "
            "Use for: flowcharts, architecture diagrams, sequence diagrams, "
            "entity-relationship diagrams, mind maps."
        ),
        params=[
            ToolParam("definition", "string", "Mermaid or Graphviz DOT diagram definition"),
            ToolParam("format", "string", "'mermaid' or 'graphviz' (default: mermaid)", required=False, default="mermaid"),
            ToolParam("filename", "string", "Output filename without extension", required=False, default="diagram"),
        ],
        allowed_departments={"design", "implementation", "documentation", "research"},
    ),

    ToolDefinition(
        name="create_chart",
        description=(
            "Create a data chart (bar, line, pie, scatter, etc.) from provided data. "
            "Saves as PNG to the project workspace. Use for: data visualisation, "
            "metrics, comparisons, trend analysis."
        ),
        params=[
            ToolParam("chart_type", "string", "'bar', 'line', 'pie', 'scatter', 'histogram', 'heatmap'"),
            ToolParam("data", "object", "Data object: {labels: [...], datasets: [{label, values: [...]}]}"),
            ToolParam("title", "string", "Chart title"),
            ToolParam("filename", "string", "Output filename without extension", required=False, default="chart"),
        ],
        allowed_departments={"research", "design", "documentation", "qa", "implementation"},
    ),

    # ── File creation ─────────────────────────────────────────────────────

    ToolDefinition(
        name="create_spreadsheet",
        description=(
            "Create an Excel spreadsheet (.xlsx) with one or more sheets of data. "
            "Saves to the project workspace. Use for: structured data output, "
            "reports with tables, datasets, comparison matrices."
        ),
        params=[
            ToolParam("sheets", "array", "List of sheet objects: [{name, headers: [...], rows: [[...]]}]"),
            ToolParam("filename", "string", "Output filename without .xlsx extension"),
        ],
        allowed_departments={"research", "documentation", "implementation", "qa"},
    ),

    ToolDefinition(
        name="create_document",
        description=(
            "Create a formatted text document (.md or .txt) with sections and content. "
            "Saves to the project workspace. Use for: reports, specifications, "
            "READMEs, documentation, meeting notes."
        ),
        params=[
            ToolParam("content", "string", "Full document content (Markdown supported)"),
            ToolParam("filename", "string", "Output filename without extension"),
            ToolParam("format", "string", "'md' or 'txt' (default: md)", required=False, default="md"),
        ],
        allowed_departments={"*"},
    ),

    ToolDefinition(
        name="read_file",
        description=(
            "Read the content of a file from the project workspace. "
            "Returns text content or metadata for binary files. "
            "Use to read files created by previous tasks or uploaded by users."
        ),
        params=[
            ToolParam("path", "string", "File path relative to project workspace root"),
            ToolParam("max_chars", "number", "Max characters to return (default 10000)", required=False, default=10000),
        ],
        allowed_departments={"*"},
    ),

    # ── Audio ─────────────────────────────────────────────────────────────

    ToolDefinition(
        name="text_to_speech",
        description=(
            "Convert text to speech and save as an audio file (.mp3). "
            "Saves to the project workspace. Use for: voiceovers, narration, "
            "audio previews, accessibility outputs."
        ),
        params=[
            ToolParam("text", "string", "Text to convert to speech"),
            ToolParam("voice", "string", "Voice style: 'neutral', 'male', 'female', 'british', 'american'", required=False, default="neutral"),
            ToolParam("filename", "string", "Output filename without .mp3 extension", required=False, default="audio"),
        ],
        allowed_departments={"design", "documentation", "implementation"},
    ),
]

# Name → definition lookup
_TOOL_MAP: dict[str, ToolDefinition] = {t.name: t for t in _ALL_TOOLS}


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

def get_tools_for_department(department: str) -> list[ToolDefinition]:
    """Return tools accessible to a department."""
    tools = []
    for tool in _ALL_TOOLS:
        if "*" in tool.allowed_departments or department in tool.allowed_departments:
            tools.append(tool)
    return tools


def get_tool(name: str) -> ToolDefinition | None:
    return _TOOL_MAP.get(name)


def build_tools_prompt(department: str) -> str:
    """Build the tools section of the system prompt for a department."""
    tools = get_tools_for_department(department)
    if not tools:
        return ""

    lines = [
        "## Available Tools\n",
        "You can use tools to gather information, create files, or execute code.",
        "To use a tool, include a tool_calls list in your JSON response:\n",
        '```json',
        '"tool_calls": [',
        '  {"tool": "web_search", "params": {"query": "your search query"}}',
        ']',
        '```\n',
        "The tool results will be returned to you before you produce your final answer.",
        "You can chain multiple tool calls — results from earlier calls are available for later ones.\n",
        "### Tools available to you:\n",
    ]
    for tool in tools:
        lines.append(tool.to_prompt_block())
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatcher — calls the right module
# ---------------------------------------------------------------------------

async def execute_tool(
    tool_name: str,
    params: dict[str, Any],
    project_id: str,
    task_id: str,
    department: str,
) -> dict[str, Any]:
    """
    Execute a tool call and return its result.

    Returns:
        {
            "tool": tool_name,
            "success": bool,
            "result": any,        # tool-specific output
            "error": str,         # only if success=False
            "files_created": [],  # relative paths of any files saved
        }
    """
    tool_def = get_tool(tool_name)
    if not tool_def:
        return {
            "tool": tool_name, "success": False,
            "error": f"Unknown tool: {tool_name}",
            "files_created": [],
        }

    # Access check
    if "*" not in tool_def.allowed_departments and department not in tool_def.allowed_departments:
        return {
            "tool": tool_name, "success": False,
            "error": f"Department '{department}' does not have access to tool '{tool_name}'",
            "files_created": [],
        }

    log.info("tool.execute", tool=tool_name, project_id=project_id, task_id=task_id, department=department)

    try:
        if tool_name == "web_search":
            from orchestrator.tools.web import web_search
            return await web_search(project_id=project_id, **params)

        elif tool_name == "fetch_url":
            from orchestrator.tools.web import fetch_url
            return await fetch_url(project_id=project_id, **params)

        elif tool_name == "execute_code":
            from orchestrator.tools.code_exec import execute_code
            return await execute_code(project_id=project_id, task_id=task_id, **params)

        elif tool_name == "generate_image":
            from orchestrator.tools.image import generate_image
            return await generate_image(project_id=project_id, **params)

        elif tool_name == "create_diagram":
            from orchestrator.tools.image import create_diagram
            return await create_diagram(project_id=project_id, **params)

        elif tool_name == "create_chart":
            from orchestrator.tools.image import create_chart
            return await create_chart(project_id=project_id, **params)

        elif tool_name == "create_spreadsheet":
            from orchestrator.tools.files import create_spreadsheet
            return await create_spreadsheet(project_id=project_id, **params)

        elif tool_name == "create_document":
            from orchestrator.tools.files import create_document
            return await create_document(project_id=project_id, **params)

        elif tool_name == "read_file":
            from orchestrator.tools.files import read_file
            return await read_file(project_id=project_id, **params)

        elif tool_name == "text_to_speech":
            from orchestrator.tools.audio import text_to_speech
            return await text_to_speech(project_id=project_id, **params)

        else:
            return {
                "tool": tool_name, "success": False,
                "error": f"Tool '{tool_name}' is defined but not yet implemented.",
                "files_created": [],
            }

    except Exception as exc:
        log.error("tool.error", tool=tool_name, project_id=project_id, error=str(exc))
        return {
            "tool": tool_name, "success": False,
            "error": f"Tool execution failed: {exc}",
            "files_created": [],
        }
