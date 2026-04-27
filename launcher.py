"""
launcher.py — Desktop entry point for the Autonomous AI Engine.

Replaces `main.py` as the application entry point when running as a
desktop app.  Starts the FastAPI backend in a background thread, waits
for it to be ready, then opens a native OS window via pywebview.

Usage (dev):
    python launcher.py

Usage (packaged):
    ./AutonomousEngine.exe          (Windows)
    ./AutonomousEngine              (Linux)
    open AutonomousEngine.app       (macOS)

The window is a native OS webview — WebView2 on Windows, WKWebView on
macOS, GTK WebKit on Linux — not a browser tab.  It talks to the local
FastAPI server on a random free port, exactly as before.

PyInstaller entry:
    pyinstaller autonomous_engine.spec
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# PyInstaller path fixup — must happen before any project imports
# ---------------------------------------------------------------------------
# When frozen by PyInstaller, __file__ points into the temp _MEIPASS dir.
# We add it to sys.path so relative imports still work.
if getattr(sys, "frozen", False):
    _BUNDLE = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    sys.path.insert(0, str(_BUNDLE))
else:
    _BUNDLE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Find a free port
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Bind to port 0 and let the OS choose a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_PORT = _free_port()
_HOST = "127.0.0.1"
_ORIGIN = f"http://{_HOST}:{_PORT}"


# ---------------------------------------------------------------------------
# FastAPI server thread
# ---------------------------------------------------------------------------

def _start_server() -> None:
    """
    Run the FastAPI app with uvicorn in the current thread.
    Called from a daemon thread so it dies when the main process exits.
    """
    import uvicorn
    from main import create_app

    app = create_app()

    uvicorn.run(
        app,
        host=_HOST,
        port=_PORT,
        log_level="warning",   # quieter in desktop mode
        access_log=False,
    )


def _wait_for_server(timeout: float = 30.0) -> bool:
    """Poll until the server is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((_HOST, _PORT), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# Window title / icon helpers
# ---------------------------------------------------------------------------

def _icon_path() -> str | None:
    """Return path to app icon if it exists."""
    for name in ("icon.ico", "icon.png", "icon.icns"):
        p = _BUNDLE / name
        if p.exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[launcher] Starting Autonomous AI Engine on {_ORIGIN}")

    # Start FastAPI in a background daemon thread
    server_thread = threading.Thread(target=_start_server, daemon=True, name="fastapi")
    server_thread.start()

    print("[launcher] Waiting for backend...")
    if not _wait_for_server(timeout=40.0):
        # If pywebview isn't available, fall back to opening a browser tab
        print("[launcher] ERROR: backend did not start in time")
        sys.exit(1)

    print("[launcher] Backend ready — opening window")

    # Try pywebview first (native window), fall back to browser
    try:
        import webview  # type: ignore

        icon = _icon_path()

        window = webview.create_window(
            title="Autonomous AI Engine",
            url=f"{_ORIGIN}/ui/",
            width=1400,
            height=860,
            min_size=(900, 600),
            background_color="#070a0f",  # matches --bg in index.html
        )

        webview.start(
            debug=False,
            icon=icon,
            # On Windows, use edgechromium (WebView2) for best compatibility
            # pywebview auto-selects the best backend available
        )

    except ImportError:
        # pywebview not installed — open in the system browser instead
        # This is a graceful fallback for development without pywebview
        print("[launcher] pywebview not found — opening in browser instead")
        print(f"[launcher] App available at: {_ORIGIN}/ui/")
        webbrowser.open(f"{_ORIGIN}/ui/")

        # Keep the server alive until Ctrl+C
        try:
            server_thread.join()
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down")

    print("[launcher] Window closed — exiting")


if __name__ == "__main__":
    main()
