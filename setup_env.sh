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
# Search common locations for Python 3.12.
PYTHON=""
CANDIDATES=(
    python3.12
    /opt/homebrew/bin/python3.12
    /opt/homebrew/Caskroom/miniforge/base/bin/python3.12
    /usr/local/bin/python3.12
    "$HOME/miniforge3/bin/python3.12"
    "$HOME/miniconda3/bin/python3.12"
    "$HOME/anaconda3/bin/python3.12"
    python3
)
for candidate in "${CANDIDATES[@]}"; do
    if [ -x "$candidate" ] || command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [ "$ver" = "3.12" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.12 is required (NDIF server constraint)."
    echo "Install it via:"
    echo "  brew install python@3.12"
    echo "  conda create -n py312 python=3.12 && conda activate py312"
    echo "  pyenv install 3.12"
    exit 1
fi
echo "Using $PYTHON ($($PYTHON --version))"

# Create venv if it doesn't exist or uses wrong Python version
if [ -d "$VENV_DIR" ]; then
    VENV_VER=$("$VENV_DIR/bin/python3" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [ "$VENV_VER" != "3.12" ]; then
        echo "Existing venv uses Python $VENV_VER, need 3.12. Recreating..."
        rm -rf "$VENV_DIR"
    else
        echo "Virtual environment already exists at $VENV_DIR (Python $VENV_VER)"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
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
    python-dotenv \
    jupyter \
    ipykernel

# Register Jupyter kernel
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name puns --display-name "Puns (3.12)"

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source venv/bin/activate"
echo ""
echo "To run Jupyter:  jupyter notebook interactive_explore.ipynb"
echo ""
echo "Make sure .env.local exists with your API keys:"
echo "  TOGETHER_API_KEY=..."
echo "  NDIF_API_KEY=..."
echo "  HF_TOKEN=..."
echo "  ANTHROPIC_API_KEY=..."
