#!/bin/bash
#SBATCH --job-name=geneagnt_api
#SBATCH --output=logs/geneagent_api_%A_%a.out
#SBATCH --error=logs/geneagent_api_%A_%a.err
#SBATCH --array=0-23                 # API models (1) * parts (24) = 24 tasks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --partition=moore


# API-based LLMs (no GPU required)
LLMS=(
    "azure-gpt-5.4-mini"
)
JUDGE_LLM="azure-claude-haiku-4-5"
# Configuration
LLMS_LEN=${#LLMS[@]}
PARTS=($(seq 1 24))
PARTS_LEN=${#PARTS[@]}

LLM=${LLMS[$((SLURM_ARRAY_TASK_ID / PARTS_LEN))]}
PART=${PARTS[$((SLURM_ARRAY_TASK_ID % PARTS_LEN))]}
INPUT_FILE="Datasets/AlzKB/sampled_noise_part${PART}.csv"
BASE_DIR="/home/lib/GeneAgent"

# Setup
cd $BASE_DIR || exit 1
mkdir -p logs
source .venv/bin/activate

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID (Task: $SLURM_ARRAY_TASK_ID)"
echo "LLM: $LLM"
echo "Input: $INPUT_FILE (part $PART / ${PARTS_LEN})"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Step 1: Run GeneAgent evaluation
echo -e "\n=== Step 1: Running GeneAgent ==="
python run_geneagent.py \
    --input "$INPUT_FILE" \
    --llm "$LLM" \
    --skip-columns full_set reduced_set \
    || exit 1

# Step 2: Evaluate predictions
echo -e "\n=== Step 2: Evaluating Predictions ==="
echo -e "\n--- Process Names ---"
python evaluate_pathway_predictions.py \
    --input "$INPUT_FILE" \
    --llm "$LLM" \
    --judge-llm "$JUDGE_LLM" \
    || exit 1

echo -e "\n--- Analytic Narratives ---"
python evaluate_pathway_predictions.py \
    --input "$INPUT_FILE" \
    --llm "$LLM" \
    --include-descriptions \
    --judge-llm "$JUDGE_LLM" \
    || exit 1

echo -e "\n=========================================="
echo "COMPLETED: $LLM"
echo "End: $(date)"
echo "Results: Outputs/${LLM}/"
echo "=========================================="
