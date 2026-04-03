#!/bin/bash
# Download and convert real agent trace data for the diagnosis demo.
# Requires: pip install datasets
#
# This creates two datasets:
#   scenario1/ — SWE-agent (Llama-70B) + Claude Code eval harness
#   scenario2/ — CoderForge (Qwen-32B) + OpenHands (various)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

mkdir -p "$DATA_DIR/scenario1" "$DATA_DIR/scenario2"

echo "Converting SWE-agent trajectories (500 runs)..."
python "$REPO_DIR/scripts/convert_swe_agent.py" "$DATA_DIR/scenario1" --count 500

echo "Converting eval harness runs..."
if [ -d "$REPO_DIR/examples/eval_harness" ]; then
    cp "$REPO_DIR/examples/eval_harness"/*.json "$DATA_DIR/scenario1/"
    echo "  copied $(ls "$REPO_DIR/examples/eval_harness"/*.json | wc -l | tr -d ' ') eval harness runs"
else
    echo "  warning: examples/eval_harness not found, skipping"
fi

echo "Converting CoderForge trajectories (1000 runs)..."
python "$REPO_DIR/scripts/convert_coderforge.py" "$DATA_DIR/scenario2" --count 1000 --split all

echo "Converting OpenHands trajectories (500 runs)..."
python "$REPO_DIR/scripts/convert_openhands.py" "$DATA_DIR/scenario2" --count 500

echo ""
echo "Data ready:"
echo "  scenario1: $(ls "$DATA_DIR/scenario1"/*.json | wc -l | tr -d ' ') runs"
echo "  scenario2: $(ls "$DATA_DIR/scenario2"/*.json | wc -l | tr -d ' ') runs"
