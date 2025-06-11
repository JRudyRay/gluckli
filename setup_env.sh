#!/bin/bash

# Create venv in a directory named .venv
python3 -m venv gluckli_env
source gluckli_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

echo "✅ Environment setup complete. Activate it with: source .venv/bin/activate"
