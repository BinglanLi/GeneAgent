#!/bin/bash
#SBATCH --job-name=geneagent_gpu
#SBATCH --output=logs/geneagent_gpu_%A_%a.out
#SBATCH --error=logs/geneagent_gpu_%A_%a.err
#SBATCH --array=1-2                    # GPU models (GPU required)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu                # GPU partition
#SBATCH --gres=gpu:1                   # Request 1 GPU


echo "Load modules..."
module load gcc
module load cuda12.6

# GPU-based LLMs (Ollama models)
# Recommended models for complex reasoning tasks:
# - gpt-oss:20b: OpenAI's GPT-3.5 equivalent model
# - qwen2.5:14b: Best balance of quality and stability
# - deepseek-r1:14b: Best reasoning capabilities``
# - llama3.3:70b: Highest quality (requires 40GB+ VRAM)
LLMS=(
    "gpt-oss:20b"
    "qwen2.5:14b"
    "llama3.3:70b"
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

# Configure Ollama for stability
export OLLAMA_KEEP_ALIVE=-1              # Keep models loaded in memory
export OLLAMA_MAX_LOADED_MODELS=1        # Load only one model at a time
export OLLAMA_NUM_PARALLEL=1             # Limit parallel requests
export OLLAMA_MAX_QUEUE=512              # Increase request queue size

# Start Ollama server if needed
echo -e "\n=== Starting Ollama Server ==="
~/ollama/ollama-manager.sh start

# Step 1: Run GeneAgent evaluation
echo -e "\n=== Step 1: Running GeneAgent ==="
python run_geneagent.py \
    --input "$INPUT_FILE" \
    --llm "$LLM" \
    || exit 1
# Stop Ollama server
echo -e "\n=== Stopping Ollama Server ==="
~/ollama/ollama-manager.sh stop

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
