#!/usr/bin/env bash
# build.sh — Build the Autonomous Engine desktop app on Linux / macOS
#
# Requirements:
#   pip install pywebview pyinstaller
#   Linux: sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0
#          gir1.2-webkit2-4.0  (or webkit2-4.1 on newer distros)
#   macOS: pywebview uses WKWebView — no extra install needed
#
# Output:
#   Linux:  dist/AutonomousEngine/AutonomousEngine
#   macOS:  dist/AutonomousEngine.app

set -euo pipefail

echo ""
echo "====================================================="
echo " Autonomous AI Engine — Desktop Build (Unix)"
echo "====================================================="
echo ""

# Detect OS
OS="$(uname -s)"

# Install system deps on Linux
if [[ "$OS" == "Linux" ]]; then
    echo "[0/4] Checking Linux WebKit dependencies..."
    if ! python3 -c "import gi" &>/dev/null; then
        echo "  Installing GTK/WebKit bindings..."
        sudo apt-get install -y \
            python3-gi python3-gi-cairo gir1.2-gtk-3.0 \
            gir1.2-webkit2-4.0 2>/dev/null \
        || sudo apt-get install -y \
            python3-gi python3-gi-cairo gir1.2-gtk-3.0 \
            gir1.2-webkit2-4.1 2>/dev/null \
        || echo "  Warning: could not install WebKit — install manually if build fails"
    else
        echo "  GTK/WebKit bindings found"
    fi
fi

echo "[1/4] Installing Python build dependencies..."
pip install pywebview pyinstaller --quiet --upgrade

echo "[2/4] Cleaning previous build..."
rm -rf build dist/AutonomousEngine dist/AutonomousEngine.app 2>/dev/null || true

echo "[3/4] Building executable (this takes 2-5 minutes)..."
pyinstaller autonomous_engine.spec --noconfirm

echo "[4/4] Finalising..."
DIST_DIR="dist/AutonomousEngine"
if [[ "$OS" == "Darwin" ]]; then
    DIST_DIR="dist/AutonomousEngine.app"
fi

# Copy env.example if no .env present
if [[ ! -f "$DIST_DIR/.env" ]] && [[ -f "env.example" ]]; then
    cp env.example "$DIST_DIR/.env.example"
    echo "  NOTE: Copy $DIST_DIR/.env.example to $DIST_DIR/.env and add your API keys"
fi

# Mark executable
if [[ "$OS" == "Linux" ]]; then
    chmod +x dist/AutonomousEngine/AutonomousEngine 2>/dev/null || true
fi

echo ""
echo "====================================================="
echo " Build complete!"
if [[ "$OS" == "Darwin" ]]; then
echo " App bundle: dist/AutonomousEngine.app"
echo " To run:     open dist/AutonomousEngine.app"
echo " To ship:    zip -r AutonomousEngine.zip dist/AutonomousEngine.app"
else
echo " Executable: dist/AutonomousEngine/AutonomousEngine"
echo " To run:     ./dist/AutonomousEngine/AutonomousEngine"
echo " To ship:    tar -czf AutonomousEngine.tar.gz dist/AutonomousEngine/"
fi
echo "====================================================="
echo ""
