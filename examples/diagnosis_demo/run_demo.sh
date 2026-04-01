#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMO_DIR="/tmp/moirai_diagnosis_demo"
CAUSES="$SCRIPT_DIR/causes.json"

rm -rf "$DEMO_DIR"
mkdir -p "$DEMO_DIR"

echo "========================================"
echo " SCENARIO 1: The Invisible Regression"
echo " (prompt change — same pass rate, different strategy)"
echo "========================================"
echo

python "$SCRIPT_DIR/generate.py" prompt_regression "$DEMO_DIR/prompt" --seed 42
echo

echo "--- Pass-rate comparison (looks fine!) ---"
moirai diff "$DEMO_DIR/prompt" --a variant=baseline --b variant=current
echo

echo "--- Trajectory evidence (reveals the problem) ---"
moirai evidence "$DEMO_DIR/prompt" \
    --baseline variant=baseline --current variant=current
echo

echo "--- Diagnosis (ranks causes) ---"
moirai diagnose "$DEMO_DIR/prompt" \
    --baseline variant=baseline --current variant=current \
    --causes "$CAUSES"
echo

echo "========================================"
echo " SCENARIO 2: Timeout Regression"
echo " (negative control — tool errors dominate)"
echo "========================================"
echo

python "$SCRIPT_DIR/generate.py" timeout_regression "$DEMO_DIR/timeout" --seed 42
echo

echo "--- Diagnosis ---"
moirai diagnose "$DEMO_DIR/timeout" \
    --baseline variant=baseline --current variant=current \
    --causes "$CAUSES"
echo

echo "========================================"
echo " ROBUSTNESS: 3 seeds, ranking must be stable"
echo "========================================"
echo

stable=true
for seed in 42 123 456; do
    python "$SCRIPT_DIR/generate.py" prompt_regression "$DEMO_DIR/seed_$seed" --seed "$seed" >/dev/null
    top=$(moirai diagnose "$DEMO_DIR/seed_$seed" \
        --baseline variant=baseline --current variant=current \
        --causes "$CAUSES" --json \
        | python -c "import sys,json; d=json.load(sys.stdin); print(d['cause_scores'][0]['cause_id'])")
    echo "  seed $seed: top cause = $top"
    if [ "$top" != "C1" ]; then
        stable=false
    fi
done

echo
if [ "$stable" = true ]; then
    echo "PASS: Ranking stable across all 3 seeds (C1 first)"
else
    echo "FAIL: Ranking not stable across seeds"
    exit 1
fi
