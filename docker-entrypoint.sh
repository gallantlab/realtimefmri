#!/usr/bin/env bash
set -e

pip3 install -e .
python3 -c "import realtimefmri"

echo "Starting realtimefmri..."
realtimefmri web_interface
