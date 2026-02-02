#!/bin/bash
# Create and configure a Python virtual environment for the puns project.
#
# Usage:
#   bash setup_env.sh
#   source venv/bin/activate

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=== Setting up puns project environment ==="

# NDIF requires Python 3.12 (must match server version).
# Try python3.12, fall back to python3 if not found.
PYTHON=""
for candidate in python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        if [ "$ver" = "3.12" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.12 is required (NDIF server constraint)."
    echo "Install it via: brew install python@3.12  (or conda/pyenv)"
    exit 1
fi
echo "Using $PYTHON ($($PYTHON --version))"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install \
    nnsight==0.5.15 \
    torch \
    numpy \
    matplotlib \
    scikit-learn \
    requests \
    python-dotenv

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source venv/bin/activate"
echo ""
echo "Make sure .env.local exists with your API keys:"
echo "  TOGETHER_API_KEY=..."
echo "  NDIF_API_KEY=..."
echo "  HF_TOKEN=..."
echo "  ANTHROPIC_API_KEY=..."
