#!/bin/bash
#SBATCH --job-name=aggregate
#SBATCH --output=logs/aggregate_%A_%a.out
#SBATCH --error=logs/aggregate_%A_%a.err
#SBATCH --array=0              # One task per GPU LLM (3 models)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# LLMs
LLMS=(
    "azure-gpt-5.4-mini"
)
JUDGE_LLM="azure-claude-haiku-4-5"

LLM=${LLMS[$SLURM_ARRAY_TASK_ID]}
BASE_DIR="/home/lib/GeneAgent"

cd $BASE_DIR || exit 1
mkdir -p logs
source .venv/bin/activate

# Collect all 24 input files for this LLM
INPUT_FILES=()
for PART in $(seq 1 24); do
    INPUT_FILES+=("Datasets/AlzKB/sampled_noise_part${PART}.csv")
done

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID (Task: $SLURM_ARRAY_TASK_ID)"
echo "LLM: $LLM"
echo "Judge LLM: $JUDGE_LLM"
echo "Input files: ${#INPUT_FILES[@]} parts"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

echo -e "\n--- Process Names ---"
python evaluate_pathway_predictions.py \
    --input "${INPUT_FILES[@]}" \
    --llm "$LLM" \
    --judge-llm "$JUDGE_LLM" \
    || exit 1

echo -e "\n--- Analytic Narratives ---"
python evaluate_pathway_predictions.py \
    --input "${INPUT_FILES[@]}" \
    --llm "$LLM" \
    --include-descriptions \
    --judge-llm "$JUDGE_LLM" \
    || exit 1

echo -e "\n=========================================="
echo "COMPLETED: $LLM"
echo "End: $(date)"
echo "Results: Outputs/${LLM}/"
echo "=========================================="
