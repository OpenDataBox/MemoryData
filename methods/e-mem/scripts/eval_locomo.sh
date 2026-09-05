#!/bin/bash
# Evaluation script for LoCoMo dataset

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get project root (parent of scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT" || exit 1

# Default values
CONFIG="./evaluation/locomo/config.yaml"
MODEL_ID=""
DATASET=""
RATIO=""
CONVERSATION_AUTO_SAVE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --model_id)
            MODEL_ID="--model_id $2"
            shift 2
            ;;
        --dataset)
            DATASET="--dataset $2"
            shift 2
            ;;
        --ratio)
            RATIO="--ratio $2"
            shift 2
            ;;
        --conversation_auto_save)
            CONVERSATION_AUTO_SAVE="--conversation_auto_save"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run evaluation from project root
uv run python -m evaluation.locomo.eval_locomo \
    --config "$CONFIG" \
    $MODEL_ID \
    $DATASET \
    $RATIO \
    $CONVERSATION_AUTO_SAVE

echo "Evaluation complete!"
