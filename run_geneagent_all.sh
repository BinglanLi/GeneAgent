#!/bin/bash
# Master script to submit all pathway evaluation jobs

mkdir -p logs

echo "Submitting pathway evaluation jobs..."
echo ""

# Submit API-based models (no GPU)
echo "1. Submitting API models (gpt-4o, azure-gpt-4o)..."
API_JOB=$(sbatch --parsable run_geneagent_eval_api.slurm)
echo "   Job ID: $API_JOB"

# Submit GPU-based models (with GPU)
echo "2. Submitting GPU models (gpt-oss:20b)..."
GPU_JOB=$(sbatch --parsable run_geneagent_eval_gpu.slurm)
echo "   Job ID: $GPU_JOB"

echo ""
echo "All jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/pathway_api_${API_JOB}_*.out"
echo "  tail -f logs/pathway_gpu_${GPU_JOB}_*.out"
