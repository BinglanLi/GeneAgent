#!/usr/bin/env python3
"""
Evaluate GeneAgent predictions against ground truth Pathway names.
Compares full_set and reduced_set predictions using ROUGE scores and semantic similarity (MedCPT).
"""

import pandas as pd
import re
from pathlib import Path
import argparse
import numpy as np
from rouge_score import rouge_scorer
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def process_text(text: str) -> list:
    """Extract process names from Final_Response_GeneAgent.txt file."""
    pattern = r'\([^)]*\)'
    segments = text.split('//')
    cleaned_segments = []
    
    for segment in segments:
        cleaned_segment = ''.join(char for char in segment)
        cleaned_segment = re.sub(pattern, '', cleaned_segment)
        cleaned_segment = cleaned_segment.replace('/', ' ').replace(",", " ").replace('"', "").replace("-", " ").strip()
        if cleaned_segment:
            cleaned_segments.append(cleaned_segment)
    
    return cleaned_segments


def extract_process_names(file_path: Path) -> list:
    """Extract process names from Final_Response_GeneAgent.txt."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    agent_text = ""
    with open(file_path, "r", encoding='utf-8') as agentfile:
        for line in agentfile.readlines():
            agent_text += line
    
    agent_text_processed = process_text(agent_text)
    agent_terms = []
    
    for text in agent_text_processed:
        if not text.strip():
            agent_terms.append("None")
            continue
            
        seg = text.split("\n")
        if len(seg) > 1:
            # Look for "Process: <name>" pattern
            process_match = None
            for line in seg:
                line_lower = line.lower()
                if "process:" in line_lower:
                    # Try to extract after "Process:" or "process:"
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        process_match = parts[1].strip()
                        # Remove any trailing punctuation or formatting
                        process_match = process_match.rstrip('.,;')
                        break
            
            if process_match:
                agent_terms.append(process_match)
            else:
                # Fallback: use first non-empty line if no Process: found
                first_line = next((s.strip() for s in seg if s.strip()), "None")
                agent_terms.append(first_line)
        else:
            # Single line - use it if not empty
            cleaned = text.strip()
            agent_terms.append(cleaned if cleaned else "None")
    
    return agent_terms


def cos_sim(a: Tensor, b: Tensor):
    """Compute cosine similarity between two tensors."""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def calculate_rouge_scores(reference: list, predictions: list, scorer):
    """Calculate ROUGE scores for predictions against reference."""
    metrics = ["rouge1", "rouge2", "rougeL"]
    results = []
    
    for ref, pred in zip(reference, predictions):
        scores = scorer.score(ref, pred)
        result = {}
        for metric in metrics:
            result[metric] = scores[metric].fmeasure
        results.append(result)
    
    return results


def calculate_semantic_similarity(reference: list, predictions: list, model, tokenizer):
    """Calculate semantic similarity using MedCPT."""
    scores = []
    
    for ref, pred in tqdm(zip(reference, predictions), desc="Calculating semantic similarity", total=len(reference)):
        with torch.no_grad():
            encoded = tokenizer(
                [ref, pred],
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=64,
            )
            
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            score = cos_sim(embeds[0], embeds[1])
            scores.append(score.tolist()[0])
    
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GeneAgent predictions against ground truth Pathway names",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='Datasets/AlzKB/selected_pathways_with_gene_sets2.csv',
        help='Path to input CSV file with Pathway column'
    )
    
    parser.add_argument(
        '--llm', '-l',
        type=str,
        default='gpt-4o',
        help='LLM model name used for predictions'
    )
    
    parser.add_argument(
        '--output-base', '-o',
        type=str,
        default=None,
        help='Base output directory (default: Outputs/{llm}/selected_pathways_with_gene_sets2)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file for evaluation results (default: {output_base}/evaluation_results.csv)'
    )
    
    parser.add_argument(
        '--skip-semantic',
        action='store_true',
        help='Skip semantic similarity calculation (faster)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_file = Path(args.input).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Set up output directories
    if args.output_base:
        base_output = Path(args.output_base).resolve()
    else:
        base_dir = Path(__file__).absolute().parent
        base_output = base_dir / "Outputs" / args.llm / "selected_pathways_with_gene_sets2"
    
    full_output_dir = base_output / "full_set"
    reduced_output_dir = base_output / "reduced_set"
    
    # Load ground truth
    print("Loading ground truth pathways...")
    df_truth = pd.read_csv(input_file)
    reference_pathways = df_truth['Pathway'].tolist()
    
    # Clean reference pathways (same cleaning as in evaluate.ipynb)
    reference_cleaned = []
    for pathway in reference_pathways:
        cleaned = pathway.replace('/', ' ').replace(",", " ").replace('"', "").replace("-", " ").strip()
        reference_cleaned.append(cleaned)
    
    print(f"Loaded {len(reference_cleaned)} reference pathways")
    
    # Extract predictions
    print("\nExtracting predictions from Final_Response_GeneAgent.txt files...")
    
    full_final_file = full_output_dir / "Final_Response_GeneAgent.txt"
    reduced_final_file = reduced_output_dir / "Final_Response_GeneAgent.txt"
    
    try:
        full_predictions = extract_process_names(full_final_file)
        print(f"Extracted {len(full_predictions)} predictions from full_set")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping full_set evaluation")
        full_predictions = []
    
    try:
        reduced_predictions = extract_process_names(reduced_final_file)
        print(f"Extracted {len(reduced_predictions)} predictions from reduced_set")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping reduced_set evaluation")
        reduced_predictions = []
    
    # Align predictions with reference (they should match by index)
    min_len = min(len(reference_cleaned), len(full_predictions), len(reduced_predictions))
    reference_cleaned = reference_cleaned[:min_len]
    full_predictions = full_predictions[:min_len]
    reduced_predictions = reduced_predictions[:min_len]
    
    print(f"\nEvaluating {min_len} pathway predictions...")
    
    # Calculate ROUGE scores
    print("\nCalculating ROUGE scores...")
    metrics = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    
    results = []
    
    # Evaluate full_set predictions
    if full_predictions:
        full_rouge = calculate_rouge_scores(reference_cleaned, full_predictions, scorer)
        for i, (ref, pred, rouge_scores) in enumerate(zip(reference_cleaned, full_predictions, full_rouge)):
            result = {
                "pathway_id": i,
                "reference": ref,
                "prediction_type": "full_set",
                "prediction": pred,
            }
            for metric in metrics:
                result[metric] = rouge_scores[metric]
            results.append(result)
    
    # Evaluate reduced_set predictions
    if reduced_predictions:
        reduced_rouge = calculate_rouge_scores(reference_cleaned, reduced_predictions, scorer)
        for i, (ref, pred, rouge_scores) in enumerate(zip(reference_cleaned, reduced_predictions, reduced_rouge)):
            result = {
                "pathway_id": i,
                "reference": ref,
                "prediction_type": "reduced_set",
                "prediction": pred,
            }
            for metric in metrics:
                result[metric] = rouge_scores[metric]
            results.append(result)
    
    # Calculate semantic similarity if requested
    if not args.skip_semantic:
        print("\nLoading MedCPT model for semantic similarity...")
        try:
            model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
            tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
            
            # Add semantic similarity scores
            if full_predictions:
                print("Calculating semantic similarity for full_set...")
                full_semantic = calculate_semantic_similarity(
                    reference_cleaned, full_predictions, model, tokenizer
                )
                # Add to results
                for i, score in enumerate(full_semantic):
                    idx = i  # Index in results for full_set
                    if idx < len(results) and results[idx]["prediction_type"] == "full_set":
                        results[idx]["semantic_similarity"] = score
            
            if reduced_predictions:
                print("Calculating semantic similarity for reduced_set...")
                reduced_semantic = calculate_semantic_similarity(
                    reference_cleaned, reduced_predictions, model, tokenizer
                )
                # Add to results
                full_count = len(full_predictions) if full_predictions else 0
                for i, score in enumerate(reduced_semantic):
                    idx = full_count + i  # Index in results for reduced_set
                    if idx < len(results) and results[idx]["prediction_type"] == "reduced_set":
                        results[idx]["semantic_similarity"] = score
                        
        except Exception as e:
            print(f"Warning: Could not calculate semantic similarity: {e}")
            print("Continuing without semantic similarity scores...")
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Calculate summary statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    if len(df_results) > 0:
        for pred_type in ["full_set", "reduced_set"]:
            df_type = df_results[df_results["prediction_type"] == pred_type]
            if len(df_type) > 0:
                print(f"\n{pred_type.upper()}:")
                print(f"  Number of predictions: {len(df_type)}")
                for metric in metrics:
                    if metric in df_type.columns:
                        mean_score = df_type[metric].mean()
                        print(f"  {metric}: {mean_score:.4f}")
                
                if "semantic_similarity" in df_type.columns:
                    mean_sem = df_type["semantic_similarity"].mean()
                    print(f"  Semantic Similarity (MedCPT): {mean_sem:.4f}")
        
        # Comparison
        if len(df_results[df_results["prediction_type"] == "full_set"]) > 0 and \
           len(df_results[df_results["prediction_type"] == "reduced_set"]) > 0:
            print("\nCOMPARISON:")
            df_full = df_results[df_results["prediction_type"] == "full_set"]
            df_reduced = df_results[df_results["prediction_type"] == "reduced_set"]
            
            for metric in metrics:
                if metric in df_full.columns and metric in df_reduced.columns:
                    full_mean = df_full[metric].mean()
                    reduced_mean = df_reduced[metric].mean()
                    diff = full_mean - reduced_mean
                    print(f"  {metric}: full_set={full_mean:.4f}, reduced_set={reduced_mean:.4f}, diff={diff:+.4f}")
            
            if "semantic_similarity" in df_full.columns and "semantic_similarity" in df_reduced.columns:
                full_sem = df_full["semantic_similarity"].mean()
                reduced_sem = df_reduced["semantic_similarity"].mean()
                diff_sem = full_sem - reduced_sem
                print(f"  Semantic Similarity: full_set={full_sem:.4f}, reduced_set={reduced_sem:.4f}, diff={diff_sem:+.4f}")
    
    # Save results
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = base_output / "evaluation_results.csv"
    
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

