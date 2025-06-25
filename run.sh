#!/bin/bash

set -e

PYTHON_VENV_DIR="venv"

# Create Python virtual environment if it doesn't exist
if [ ! -d "$PYTHON_VENV_DIR" ]; then
  python -m venv "$PYTHON_VENV_DIR"
fi

# Activate venv (Windows path)
source "$PYTHON_VENV_DIR/Scripts/activate"

# Install Python dependencies
pip install -r requirements.txt

# Start backend
python main.py