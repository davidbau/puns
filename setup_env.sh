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

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
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
    nnsight==0.5.11 \
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
