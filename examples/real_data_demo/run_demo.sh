#!/bin/bash
# Real-data diagnosis demo.
#
# Prerequisite: run setup_data.sh first, or point DATA_DIR at existing data.
#
# Two scenarios:
#   1. "The Cognitive Gap" — same pass rate, radically different reasoning
#   2. "The Ambiguous Regression" — multiple competing explanations
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CAUSES="$SCRIPT_DIR/causes.json"

# Use pre-converted data if available, otherwise use existing examples
if [ -d "$SCRIPT_DIR/data/scenario1" ] && [ "$(ls "$SCRIPT_DIR/data/scenario1"/*.json 2>/dev/null | wc -l)" -gt 0 ]; then
    S1_DATA="$SCRIPT_DIR/data/scenario1"
    S2_DATA="$SCRIPT_DIR/data/scenario2"
else
    echo "Pre-converted data not found. Using quick setup with existing data..."
    S1_DATA=$(mktemp -d)
    S2_DATA=$(mktemp -d)
    trap "rm -rf $S1_DATA $S2_DATA" EXIT

    # Scenario 1: pool swe_agent + eval_harness
    if [ -d "$SCRIPT_DIR/../swe_agent" ] && [ -d "$SCRIPT_DIR/../eval_harness" ]; then
        cp "$SCRIPT_DIR/../swe_agent"/*.json "$S1_DATA/"
        cp "$SCRIPT_DIR/../eval_harness"/*.json "$S1_DATA/"
    else
        echo "error: examples/swe_agent and examples/eval_harness required"
        exit 1
    fi

    # Scenario 2: need coderforge + openhands (convert small batch)
    echo "Converting CoderForge (200 runs)..."
    python "$(dirname "$SCRIPT_DIR")/../scripts/convert_coderforge.py" "$S2_DATA" --count 200 --split SWE_Rebench
    echo "Converting OpenHands (200 runs)..."
    python "$(dirname "$SCRIPT_DIR")/../scripts/convert_openhands.py" "$S2_DATA" --count 200
fi

echo ""
echo "================================================================"
echo " SCENARIO 1: The Cognitive Gap"
echo " SWE-agent (Llama-70B) vs Claude Code eval harness"
echo " Same class of tasks, similar pass rates, different everything"
echo "================================================================"
echo ""

echo "--- Aggregate comparison (pass rate looks similar) ---"
moirai diff "$S1_DATA" --a harness=swe-bench --b harness=flat_llm
echo ""

echo "--- Trajectory evidence (reveals the gap) ---"
moirai evidence "$S1_DATA" \
    --baseline harness=swe-bench --current harness=flat_llm
echo ""

echo "--- Diagnosis (what explains the behavioral difference?) ---"
moirai diagnose "$S1_DATA" \
    --baseline harness=swe-bench --current harness=flat_llm \
    --causes "$CAUSES" --bootstrap 100
echo ""

echo "================================================================"
echo " SCENARIO 2: The Ambiguous Regression"
echo " CoderForge (Qwen-32B) vs OpenHands (various models)"
echo " Similar pass rates, no single dominant explanation"
echo "================================================================"
echo ""

echo "--- Aggregate comparison ---"
moirai diff "$S2_DATA" --a harness=coderforge --b harness=swe-rebench
echo ""

echo "--- Trajectory evidence ---"
moirai evidence "$S2_DATA" \
    --baseline harness=coderforge --current harness=swe-rebench
echo ""

echo "--- Diagnosis ---"
moirai diagnose "$S2_DATA" \
    --baseline harness=coderforge --current harness=swe-rebench \
    --causes "$CAUSES" --bootstrap 100
echo ""

echo "================================================================"
echo " COMPARISON"
echo "================================================================"
echo ""
echo "Scenario 1: One dominant cause (reasoning_approach at ~0.90)"
echo "  → Clear signal. The agents think differently."
echo ""
echo "Scenario 2: Four competing causes, none dominant"
echo "  → Ambiguous. Multiple factors contribute."
echo "  → Unknown bucket at ~0.16 = honest uncertainty."
echo ""
echo "Both scenarios: pass-rate difference is <10%."
echo "Both scenarios: trajectory structure reveals what pass rate hides."
