#!/bin/sh
# Initialiser program. Sets up a venv and installs requirements.
set -e

echo "Creating the virtual environment..."
python3 -m venv venv

echo "Installing the required packages..."
venv/bin/python -m pip install -r requirements.txt

echo "\nSetup complete!"
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo "Note: you must do this at the start of every session if you want to run the code!"

echo "\nTo deactivate the environment, run:"
echo "  deactivate"