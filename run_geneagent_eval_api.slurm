#!/bin/bash
#SBATCH --job-name=geneagnt_api
#SBATCH --output=logs/geneagent_api_%A_%a.out
#SBATCH --error=logs/geneagent_api_%A_%a.err
#SBATCH --array=0                    # API models (no GPU needed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --partition=moore

# API-based LLMs (no GPU required)
LLMS=(
    "azure-gpt-4o"
)

# Configuration
INPUT_FILE="Datasets/AlzKB/selected_pathways_with_descriptions.csv"
BASE_DIR="/home/lib/GeneAgent"
LLM=${LLMS[$SLURM_ARRAY_TASK_ID]}

# Setup
cd $BASE_DIR || exit 1
mkdir -p logs
source .venv/bin/activate

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID (Task: $SLURM_ARRAY_TASK_ID)"
echo "LLM: $LLM"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Step 1: Run GeneAgent evaluation
echo -e "\n=== Step 1: Running GeneAgent ==="
python run_geneagent.py \
    --input "$INPUT_FILE" \
    --llm "$LLM" \
    --skip-full
    || exit 1

# Step 2: Evaluate predictions
echo -e "\n=== Step 2: Evaluating Predictions ==="
python evaluate_pathway_predictions.py \
    --input "$INPUT_FILE" \
    --llm "$LLM" \
    || exit 1

echo -e "\n=========================================="
echo "COMPLETED: $LLM"
echo "End: $(date)"
echo "Results: Outputs/${LLM}/"
echo "=========================================="
