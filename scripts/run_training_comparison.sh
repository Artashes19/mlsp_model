#!/bin/bash
#
# Run cross-branch training comparison between devbugfix_khoren and korean-model
#
# This script runs both training pipelines with controlled configs and compares
# the resulting model weights to verify training equivalence.
#
# Usage:
#   ./scripts/run_training_comparison.sh [precision]
#
# Arguments:
#   precision: "32" (fp32), "16" (fp16), or "bf16-mixed" (default)
#
# Examples:
#   ./scripts/run_training_comparison.sh           # Run with bf16-mixed (default)
#   ./scripts/run_training_comparison.sh 32        # Run with fp32
#   ./scripts/run_training_comparison.sh bf16-mixed # Run with bf16
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default precision
PRECISION="${1:-bf16-mixed}"

# Validate precision
if [[ "$PRECISION" != "32" && "$PRECISION" != "16" && "$PRECISION" != "bf16-mixed" ]]; then
    echo "Error: Invalid precision '$PRECISION'"
    echo "Valid options: 32, 16, bf16-mixed"
    exit 1
fi

echo "============================================================"
echo "CROSS-BRANCH TRAINING COMPARISON"
echo "============================================================"
echo ""
echo "Repository: $REPO_ROOT"
echo "Precision:  $PRECISION"
echo ""

# Check we're on devbugfix branch
CURRENT_BRANCH=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != *"devbugfix"* && "$CURRENT_BRANCH" != *"dev"* ]]; then
    echo "Warning: Expected to be on devbugfix_khoren branch, but on: $CURRENT_BRANCH"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Clean up previous results
echo "Cleaning up previous results..."
rm -rf /tmp/training_comparison/devbugfix/* /tmp/training_comparison/korean/2026-* /tmp/training_comparison/korean/data_subset 2>/dev/null || true

# Set environment variable for korean_mode
export INDOOR_OVERRIDES_CONFIG="$REPO_ROOT/configs/korean_mode.yaml"

# Run the comparison
echo ""
echo "Starting training comparison with precision=$PRECISION..."
echo ""

cd "$REPO_ROOT"
python3 tests/compare_training.py --full --precision "$PRECISION"
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo ""
echo "Precision: $PRECISION"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Result:    ✓ SUCCESS - Training pipelines are EQUIVALENT"
else
    echo "Result:    ✗ FAILED - Training pipelines differ"
fi

echo ""
echo "Detailed results in /tmp/training_comparison/"
echo "============================================================"

exit $EXIT_CODE
