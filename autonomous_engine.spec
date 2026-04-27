# autonomous_engine.spec
#
# PyInstaller spec for building the Autonomous AI Engine desktop app.
#
# Build commands:
#   Windows:  build.bat
#   Linux:    ./build.sh
#   macOS:    ./build.sh
#
# Output:
#   dist/AutonomousEngine/AutonomousEngine.exe  (Windows one-folder)
#   dist/AutonomousEngine/AutonomousEngine      (Linux)
#   dist/AutonomousEngine.app                   (macOS)
#
# One-folder mode (not one-file) is used because:
#   - chromadb, ONNX, DuckDB all have large native libraries
#   - one-file mode unpacks to a temp dir on every launch (slow, ~5s delay)
#   - one-folder can be zipped and distributed just as easily

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH)

# ---------------------------------------------------------------------------
# Collect data that must be bundled
# ---------------------------------------------------------------------------

datas = [
    # UI — the HTML/CSS/JS frontend
    (str(ROOT / "ui"), "ui"),

    # .env is NOT bundled — users supply their own
    # Skills folder
    (str(ROOT / "skills"), "skills"),
]

# chromadb ships ONNX models and migration SQL files that must be included
_chroma_d, _chroma_b, _chroma_h = collect_all("chromadb")
datas       += _chroma_d
binaries     = _chroma_b
hiddenimports = _chroma_h

# SQLAlchemy dialects
_sqla_d, _sqla_b, _sqla_h = collect_all("sqlalchemy")
datas         += _sqla_d
binaries      += _sqla_b
hiddenimports += _sqla_h

# pydantic v2 has Rust extensions
_pyd_d, _pyd_b, _pyd_h = collect_all("pydantic")
datas         += _pyd_d
binaries      += _pyd_b
hiddenimports += _pyd_h

# uvicorn + starlette internals
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("starlette")
hiddenimports += collect_submodules("fastapi")

# aiosqlite / asyncpg
hiddenimports += collect_submodules("aiosqlite")
hiddenimports += ["asyncpg", "asyncpg.pgproto.pgproto", "greenlet"]

# httpx
hiddenimports += collect_submodules("httpx")

# structlog
hiddenimports += collect_submodules("structlog")

# Our own packages
hiddenimports += [
    "api.health", "api.projects", "api.tasks", "api.audit",
    "api.learning", "api.checkpoints", "api.governance", "api.sse",
    "orchestrator.engine_orchestrator", "orchestrator.agent_runner",
    "orchestrator.planner", "orchestrator.validator",
    "orchestrator.scheduler", "orchestrator.audit",
    "orchestrator.skill_registry", "orchestrator.checkpoint_manager",
    "orchestrator.workspace_manager",
    "llm.client", "llm.tier_router", "llm.prompts", "llm.schemas",
    "llm.providers.base", "llm.providers.ollama",
    "llm.providers.openai_compat", "llm.providers.anthropic_provider",
    "models.project", "models.task",
    "memory.store",
    "core.ids",
    "db.engine",
    "config",
    "main",
    # Windows WebView2 backend for pywebview
    "webview",
    "webview.platforms.winforms",   # Windows
    "webview.platforms.cocoa",      # macOS
    "webview.platforms.gtk",        # Linux
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Things we definitely don't need
        "matplotlib", "PIL", "PIL.Image",
        "tkinter", "wx", "PyQt5", "PyQt6",
        "IPython", "jupyter", "notebook",
        "pytest", "sphinx",
        "black", "mypy", "flake8",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

# ---------------------------------------------------------------------------
# EXE
# ---------------------------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # one-folder mode
    name="AutonomousEngine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # no terminal window on Windows
    icon=str(ROOT / "icon.ico") if (ROOT / "icon.ico").exists() else None,
)

# ---------------------------------------------------------------------------
# COLLECT (one-folder bundle)
# ---------------------------------------------------------------------------

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AutonomousEngine",
)

# ---------------------------------------------------------------------------
# macOS .app bundle
# ---------------------------------------------------------------------------

app = BUNDLE(
    coll,
    name="AutonomousEngine.app",
    icon=str(ROOT / "icon.icns") if (ROOT / "icon.icns").exists() else None,
    bundle_identifier="com.autonomousengine.app",
    info_plist={
        "NSHighResolutionCapable": True,
        "CFBundleShortVersionString": "0.1.0",
        "LSMinimumSystemVersion": "12.0",
    },
)
