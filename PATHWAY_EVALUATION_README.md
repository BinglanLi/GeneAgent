# Pathway Evaluation Guide

This guide explains how to evaluate GeneAgent predictions on the `selected_pathways_with_descriptions.csv` dataset, comparing predictions from `full_set` and `reduced_set` against ground truth `Pathway` names.

## Overview

The evaluation process consists of two steps:

1. **Run GeneAgent** on both `full_set` and `reduced_set` columns
2. **Evaluate predictions** against ground truth using ROUGE scores and semantic similarity (MedCPT)

## Step 1: Run GeneAgent on Both Gene Sets

Use `run_geneagent.py` to process both gene sets:

```bash
python run_geneagent.py \
    --input Datasets/AlzKB/selected_pathways_with_descriptions.csv \
    --llm gpt-4o \
    -o /home/lib/GeneAgent/Outputs/azure-gpt-4o/ \
    --limit 1  # Optional: limit for testing

python run_geneagent.py \
    --input Datasets/AlzKB/selected_pathways_with_descriptions.csv \
    --llm azure-gpt-4o \
    -o /home/lib/GeneAgent/Outputs/azure-gpt-4o/ \
    --limit 1  # Optional: limit for testing

~/ollama/ollama-manager.sh start
python run_geneagent.py \
    --input Datasets/AlzKB/selected_pathways_with_descriptions.csv \
    --llm gpt-oss:20b \
    -o /home/lib/GeneAgent/Outputs/azure-gpt-4o/ \
    --limit 1  # Optional: limit for testing
~/ollama/ollama-manager.sh stop
```

### Options:
- `--input`: Path to the CSV file (default: `Datasets/AlzKB/selected_pathways_with_descriptions.csv`)
- `--llm`: LLM model name (default: `gpt-4o`)
- `--output-base`: Base output directory (default: `Outputs/{llm}/{input_filename_stem}`)
- `--skip-full`: Skip processing full_set if already done
- `--skip-reduced`: Skip processing reduced_set if already done
- `--limit`: Limit number of pathways to process (useful for testing)

### Output Structure:
```
Outputs/{llm}/{input_filename_stem}/
├── full_set/
│   ├── Final_Response_GeneAgent.txt
│   ├── Baseline_LLM_Responses.txt
│   └── ...
├── reduced_set/
│   ├── Final_Response_GeneAgent.txt
│   ├── Baseline_LLM_Responses.txt
│   └── ...
└── reference_pathways.csv
```

## Step 2: Evaluate Predictions

After running GeneAgent, evaluate the predictions:

```bash
python evaluate_pathway_predictions.py \
    --input Datasets/AlzKB/selected_pathways_with_descriptions.csv \
    --llm gpt-4o
```

### Options:
- `--input`: Path to the CSV file with Pathway column (default: `Datasets/AlzKB/selected_pathways_with_descriptions.csv`)
- `--llm`: LLM model name used for predictions (default: `gpt-4o`)
- `--output-base`: Base output directory (default: `Outputs/{llm}/{input_filename_stem}`)
- `--output-file`: Custom output file for results (default: `{output_base}/evaluation_results.csv`)
- `--skip-semantic`: Skip semantic similarity calculation (faster, but less complete)

### Output:
- Console output with summary statistics
- CSV file with detailed results: `evaluation_results.csv`

The evaluation results CSV contains:
- `pathway_id`: Index of the pathway
- `reference`: Ground truth pathway name
- `prediction_type`: Either "full_set" or "reduced_set"
- `prediction`: Predicted process name
- `rouge1`, `rouge2`, `rougeL`: ROUGE scores
- `semantic_similarity`: MedCPT cosine similarity (if calculated)

## Example Workflow

```bash
# 1. Run GeneAgent on both gene sets (test with 3 pathways first)
python run_geneagent.py \
    --input Datasets/AlzKB/selected_pathways_with_descriptions.csv \
    --llm gpt-4o \
    --limit 3

# 2. Evaluate the predictions
python evaluate_pathway_predictions.py \
    --input Datasets/AlzKB/selected_pathways_with_descriptions.csv \
    --llm gpt-4o

# 3. For full dataset (remove --limit)
python run_geneagent.py \
    --input Datasets/AlzKB/selected_pathways_with_descriptions.csv \
    --llm gpt-4o

# 4. Evaluate full results
python evaluate_pathway_predictions.py \
    --input Datasets/AlzKB/selected_pathways_with_descriptions.csv \
    --llm gpt-4o
```

## Resuming Processing

If processing was interrupted, you can resume:

```bash
# Resume full_set (if it was already processed)
python run_geneagent.py \
    --input Datasets/AlzKB/selected_pathways_with_descriptions.csv \
    --llm gpt-4o \
    --skip-full

# Or resume reduced_set
python run_geneagent.py \
    --input Datasets/AlzKB/selected_pathways_with_descriptions.csv \
    --llm gpt-4o \
    --skip-reduced
```

## Evaluation Metrics

The evaluation uses two types of metrics:

1. **ROUGE Scores** (from `rouge-score` package):
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: Longest common subsequence

2. **Semantic Similarity** (MedCPT):
   - Cosine similarity between embeddings of reference and prediction
   - Uses `ncbi/MedCPT-Query-Encoder` model

## Notes

- The scripts automatically parse gene lists from string representations (e.g., `"['GENE1', 'GENE2']"`)
- Predictions are extracted from `Final_Response_GeneAgent.txt` files
- Process names are extracted using the pattern `"Process: <name>"` or from the first line
- Results are aligned by index (pathway order in the CSV)

