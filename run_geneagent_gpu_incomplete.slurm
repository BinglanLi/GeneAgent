#!/bin/bash
#SBATCH --job-name=geneagent_incomplete
#SBATCH --output=logs/geneagent_incomplete_%A_%a.out
#SBATCH --error=logs/geneagent_incomplete_%A_%a.err
#SBATCH --array=0-4          # 5 tasks: mixtral:8x22b (5)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2

echo "Load modules..."
module load gcc
module load cuda12.6

# Format: "llm|part|skip_col1 skip_col2 ..."  (empty skip = all columns incomplete, run all)
# Skip columns are those already complete and should not be overwritten.
CONFIGS=(
    # mixtral:8x22b (indices 0-4)
    "mixtral:8x22b|11|full_set reduced_set noise_20 noise_40 noise_60"
    "mixtral:8x22b|13|full_set noise_20 noise_40 noise_60 noise_80"
    "mixtral:8x22b|15|reduced_set noise_20 noise_40 noise_60 noise_80"
    "mixtral:8x22b|18|full_set reduced_set noise_40 noise_60 noise_80"
    "mixtral:8x22b|20|reduced_set noise_20 noise_40 noise_60 noise_80"
)

IFS='|' read -r LLM PART SKIP_COLS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

INPUT_FILE="Datasets/AlzKB/sampled_noise_part${PART}.csv"
BASE_DIR="/home/lib/GeneAgent"

cd $BASE_DIR || exit 1
mkdir -p logs
source .venv/bin/activate

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID (Task: $SLURM_ARRAY_TASK_ID)"
echo "LLM: $LLM"
echo "Input: $INPUT_FILE (part $PART)"
echo "Skip columns: ${SKIP_COLS:-none}"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "=========================================="

# Configure Ollama for stability
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_QUEUE=512

echo -e "\n=== Starting Ollama Server ==="
~/ollama/ollama-manager.sh start

echo -e "\n=== Running GeneAgent ==="
SKIP_ARGS=()
if [ -n "$SKIP_COLS" ]; then
    SKIP_ARGS=(--skip-columns $SKIP_COLS)
fi

python run_geneagent.py \
    --input "$INPUT_FILE" \
    --llm "$LLM" \
    "${SKIP_ARGS[@]}" \
    || exit 1

echo -e "\n=== Stopping Ollama Server ==="
~/ollama/ollama-manager.sh stop

echo -e "\n=== Step 2: Evaluating Predictions ==="
python evaluate_pathway_predictions.py \
    --input "$INPUT_FILE" \
    --llm "$LLM" \
    || exit 1

echo -e "\n=========================================="
echo "COMPLETED: $LLM part${PART}"
echo "End: $(date)"
echo "Results: Outputs/${LLM}/"
echo "=========================================="
