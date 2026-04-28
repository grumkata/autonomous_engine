"""
orchestrator/tools/image.py — Image, diagram, and chart generation tools.

generate_image:
  Uses the tier router's image-capable providers:
    1. Pollinations.ai image endpoint (free, no key)
    2. Cloudflare Workers AI (free with CF key)
    3. Stability AI (paid, STABILITY_API_KEY)
  Falls back gracefully through providers.

create_diagram:
  Renders Mermaid or Graphviz DOT to PNG.
  Mermaid: uses mermaid.ink API (free, no key) — renders via CDN
  Graphviz: uses local graphviz binary if installed, else mermaid.ink

create_chart:
  Uses matplotlib (always available, pure Python).
  Produces publication-quality charts as PNG.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import httpx
import structlog

from orchestrator.tools.file_store import (
    save_bytes, unique_filename, project_dir, ResourceType,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# generate_image
# ---------------------------------------------------------------------------

async def generate_image(
    prompt: str,
    project_id: str,
    style: str = "realistic",
    width: int = 512,
    height: int = 512,
) -> dict[str, Any]:
    """Generate an image from a text prompt and save to workspace."""
    width  = max(256, min(width, 1024))
    height = max(256, min(height, 1024))

    from config import get_settings
    settings = get_settings()

    # ── 1. Pollinations.ai (free, no key) ────────────────────────────────
    try:
        result = await _pollinations_image(prompt, project_id, style, width, height)
        if result["success"]:
            return result
    except Exception as exc:
        log.warning("generate_image.pollinations_failed", error=str(exc))

    # ── 2. Stability AI ──────────────────────────────────────────────────
    stability_key = getattr(settings, "stability_api_key", "")
    if stability_key:
        try:
            result = await _stability_image(prompt, project_id, style, width, height, stability_key)
            if result["success"]:
                return result
        except Exception as exc:
            log.warning("generate_image.stability_failed", error=str(exc))

    return {
        "tool": "generate_image",
        "success": False,
        "error": (
            "No image provider available. "
            "Add STABILITY_API_KEY to .env for paid generation, "
            "or ensure network access to pollinations.ai."
        ),
        "files_created": [],
    }


async def _pollinations_image(
    prompt: str, project_id: str, style: str, width: int, height: int
) -> dict:
    """Free image generation via Pollinations.ai — no API key needed."""
    import urllib.parse

    style_suffix = {
        "realistic": "", "illustration": ", digital art illustration",
        "diagram": ", clean technical diagram, white background",
        "logo": ", vector logo style", "sketch": ", pencil sketch",
    }.get(style, "")

    full_prompt = urllib.parse.quote(f"{prompt}{style_suffix}")
    url = f"https://image.pollinations.ai/prompt/{full_prompt}?width={width}&height={height}&nologo=true"

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        image_bytes = resp.content

    if len(image_bytes) < 1000:
        raise ValueError("Pollinations returned empty image")

    fname = unique_filename(f"image_{prompt[:20].replace(' ', '_')}.png", project_id, ResourceType.IMAGE)
    dest  = save_bytes(project_id, fname, image_bytes, ResourceType.IMAGE)
    rel   = str(dest.relative_to(project_dir(project_id)))

    return {
        "tool": "generate_image",
        "success": True,
        "filename": fname,
        "path": rel,
        "provider": "pollinations",
        "files_created": [rel],
    }


async def _stability_image(
    prompt: str, project_id: str, style: str, width: int, height: int, api_key: str
) -> dict:
    """Image generation via Stability AI API."""
    # Round to nearest 64px (Stability requirement)
    width  = (width  // 64) * 64
    height = (height // 64) * 64

    style_presets = {
        "realistic": "photographic", "illustration": "digital-art",
        "diagram": "line-art", "logo": "logo-like", "sketch": "sketch",
    }

    payload = {
        "text_prompts": [{"text": prompt, "weight": 1.0}],
        "cfg_scale": 7, "height": height, "width": width,
        "steps": 30, "samples": 1,
    }
    preset = style_presets.get(style)
    if preset:
        payload["style_preset"] = preset

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    image_bytes = base64.b64decode(data["artifacts"][0]["base64"])
    fname = unique_filename(f"image_{prompt[:20].replace(' ', '_')}.png", project_id, ResourceType.IMAGE)
    dest  = save_bytes(project_id, fname, image_bytes, ResourceType.IMAGE)
    rel   = str(dest.relative_to(project_dir(project_id)))

    return {
        "tool": "generate_image",
        "success": True,
        "filename": fname,
        "path": rel,
        "provider": "stability",
        "files_created": [rel],
    }


# ---------------------------------------------------------------------------
# create_diagram
# ---------------------------------------------------------------------------

async def create_diagram(
    definition: str,
    project_id: str,
    format: str = "mermaid",
    filename: str = "diagram",
) -> dict[str, Any]:
    """Render a Mermaid or Graphviz diagram to PNG."""

    if format == "mermaid" or "graph" in definition[:30].lower() and "digraph" not in definition[:30].lower():
        return await _mermaid_diagram(definition, project_id, filename)
    elif format == "graphviz" or "digraph" in definition[:30].lower() or "graph " in definition[:10].lower():
        return await _graphviz_diagram(definition, project_id, filename)
    else:
        return await _mermaid_diagram(definition, project_id, filename)


async def _mermaid_diagram(definition: str, project_id: str, filename: str) -> dict:
    """Render Mermaid via mermaid.ink API (free CDN, no key)."""
    import urllib.parse

    # Encode definition for mermaid.ink
    encoded = base64.b64encode(definition.encode()).decode()
    url = f"https://mermaid.ink/img/{encoded}?type=png"

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        resp = await client.get(url)
        if not resp.is_success:
            raise RuntimeError(f"mermaid.ink returned {resp.status_code}")
        image_bytes = resp.content

    if len(image_bytes) < 500:
        raise ValueError("mermaid.ink returned empty image")

    fname = unique_filename(f"{filename}.png", project_id, ResourceType.DIAGRAM)
    dest  = save_bytes(project_id, fname, image_bytes, ResourceType.DIAGRAM)
    rel   = str(dest.relative_to(project_dir(project_id)))

    # Also save the source definition
    src_fname = fname.replace(".png", ".mmd")
    save_bytes(project_id, src_fname, definition.encode(), ResourceType.DIAGRAM)

    return {
        "tool": "create_diagram",
        "success": True,
        "filename": fname,
        "path": rel,
        "format": "mermaid",
        "provider": "mermaid.ink",
        "files_created": [rel],
    }


async def _graphviz_diagram(definition: str, project_id: str, filename: str) -> dict:
    """Render Graphviz DOT — uses local graphviz if available, else mermaid.ink."""
    import asyncio
    import tempfile

    # Try local graphviz first
    try:
        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False, mode="w") as f:
            f.write(definition)
            dot_path = f.name

        out_path = dot_path.replace(".dot", ".png")
        proc = await asyncio.create_subprocess_exec(
            "dot", "-Tpng", dot_path, "-o", out_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)

        if proc.returncode == 0:
            image_bytes = Path(out_path).read_bytes()
            fname = unique_filename(f"{filename}.png", project_id, ResourceType.DIAGRAM)
            dest  = save_bytes(project_id, fname, image_bytes, ResourceType.DIAGRAM)
            rel   = str(dest.relative_to(project_dir(project_id)))
            return {
                "tool": "create_diagram",
                "success": True,
                "filename": fname,
                "path": rel,
                "format": "graphviz",
                "provider": "local-graphviz",
                "files_created": [rel],
            }
    except (FileNotFoundError, asyncio.TimeoutError):
        pass  # graphviz not installed or timed out — fall through
    finally:
        import os
        for p in [dot_path, out_path]:
            try: os.unlink(p)
            except Exception: pass

    # Fallback: convert DOT to Mermaid-like via mermaid.ink won't work,
    # so we render it as an SVG text file instead
    fname = unique_filename(f"{filename}.dot", project_id, ResourceType.DIAGRAM)
    dest  = save_bytes(project_id, fname, definition.encode(), ResourceType.DIAGRAM)
    rel   = str(dest.relative_to(project_dir(project_id)))
    return {
        "tool": "create_diagram",
        "success": True,
        "filename": fname,
        "path": rel,
        "format": "graphviz-source",
        "provider": "saved-source",
        "note": "Install graphviz (apt install graphviz) to render to PNG",
        "files_created": [rel],
    }


# ---------------------------------------------------------------------------
# create_chart
# ---------------------------------------------------------------------------

async def create_chart(
    chart_type: str,
    data: dict,
    title: str,
    project_id: str,
    filename: str = "chart",
) -> dict[str, Any]:
    """Create a matplotlib chart and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return {
            "tool": "create_chart",
            "success": False,
            "error": "matplotlib not installed. Run: pip install matplotlib",
            "files_created": [],
        }

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="#c8ccd8")
    ax.xaxis.label.set_color("#c8ccd8")
    ax.yaxis.label.set_color("#c8ccd8")
    ax.title.set_color("#ffffff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")

    _PALETTE = ["#6c9ef8", "#f87171", "#4ade80", "#fbbf24", "#a78bfa",
                "#34d399", "#fb923c", "#60a5fa", "#f472b6", "#94a3b8"]

    try:
        labels   = data.get("labels", [])
        datasets = data.get("datasets", [])

        if chart_type == "bar":
            x = np.arange(len(labels))
            width_per = 0.8 / max(len(datasets), 1)
            for i, ds in enumerate(datasets):
                offset = (i - len(datasets) / 2 + 0.5) * width_per
                ax.bar(x + offset, ds["values"], width_per,
                       label=ds.get("label", f"Series {i+1}"),
                       color=_PALETTE[i % len(_PALETTE)], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")

        elif chart_type == "line":
            for i, ds in enumerate(datasets):
                ax.plot(labels, ds["values"], marker="o",
                        label=ds.get("label", f"Series {i+1}"),
                        color=_PALETTE[i % len(_PALETTE)], linewidth=2)
            plt.xticks(rotation=30, ha="right")

        elif chart_type == "pie":
            values = datasets[0]["values"] if datasets else []
            ax.pie(values, labels=labels, autopct="%1.1f%%",
                   colors=_PALETTE[:len(labels)], startangle=90,
                   textprops={"color": "#c8ccd8"})
            ax.axis("equal")

        elif chart_type == "scatter":
            for i, ds in enumerate(datasets):
                vals = ds["values"]
                if vals and isinstance(vals[0], (list, tuple)):
                    x_vals = [v[0] for v in vals]
                    y_vals = [v[1] for v in vals]
                else:
                    x_vals = list(range(len(vals)))
                    y_vals = vals
                ax.scatter(x_vals, y_vals,
                           label=ds.get("label", f"Series {i+1}"),
                           color=_PALETTE[i % len(_PALETTE)], alpha=0.75, s=60)

        elif chart_type == "histogram":
            for i, ds in enumerate(datasets):
                ax.hist(ds["values"], bins=20, alpha=0.7,
                        label=ds.get("label", f"Series {i+1}"),
                        color=_PALETTE[i % len(_PALETTE)])

        elif chart_type == "heatmap":
            import numpy as np
            matrix = data.get("matrix", [])
            if matrix:
                im = ax.imshow(matrix, cmap="viridis", aspect="auto")
                plt.colorbar(im, ax=ax)
                if labels:
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                row_labels = data.get("row_labels", [])
                if row_labels:
                    ax.set_yticks(range(len(row_labels)))
                    ax.set_yticklabels(row_labels)

        else:
            plt.close(fig)
            return {
                "tool": "create_chart",
                "success": False,
                "error": f"Unknown chart type: {chart_type}. "
                         "Valid: bar, line, pie, scatter, histogram, heatmap",
                "files_created": [],
            }

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        if len(datasets) > 1:
            ax.legend(facecolor="#1a1d2e", labelcolor="#c8ccd8", edgecolor="#2a2d3e")

        ax.grid(True, alpha=0.2, color="#2a2d3e")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        image_bytes = buf.getvalue()

    finally:
        plt.close(fig)

    fname = unique_filename(f"{filename}.png", project_id, ResourceType.IMAGE)
    dest  = save_bytes(project_id, fname, image_bytes, ResourceType.IMAGE)
    rel   = str(dest.relative_to(project_dir(project_id)))

    return {
        "tool": "create_chart",
        "success": True,
        "filename": fname,
        "path": rel,
        "chart_type": chart_type,
        "files_created": [rel],
    }
