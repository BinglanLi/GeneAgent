#!/bin/bash
#SBATCH --job-name=geneagent_gpu
#SBATCH --output=logs/geneagent_gpu_%A_%a.out
#SBATCH --error=logs/geneagent_gpu_%A_%a.err
#SBATCH --array=0                      # GPU models (GPU required)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu                # GPU partition
#SBATCH --gres=gpu:1                   # Request 1 GPU

# GPU-based LLMs (Ollama models)
LLMS=(
    "gpt-oss:20b"
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
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "=========================================="

# Start Ollama server if needed
echo -e "\n=== Starting Ollama Server ==="
~/ollama/ollama-manager.sh start

# Step 1: Run GeneAgent evaluation
echo -e "\n=== Step 1: Running GeneAgent ==="
python run_pathway_evaluation.py \
    --input "$INPUT_FILE" \
    --llm "$LLM" \
    || { ~/ollama/ollama-manager.sh stop; exit 1; }

# Step 2: Evaluate predictions
echo -e "\n=== Step 2: Evaluating Predictions ==="
python evaluate_pathway_predictions.py \
    --input "$INPUT_FILE" \
    --llm "$LLM" \
    || { ~/ollama/ollama-manager.sh stop; exit 1; }

# Stop Ollama server
echo -e "\n=== Stopping Ollama Server ==="
~/ollama/ollama-manager.sh stop

echo -e "\n=========================================="
echo "COMPLETED: $LLM"
echo "End: $(date)"
echo "Results: Outputs/${LLM}/"
echo "=========================================="
