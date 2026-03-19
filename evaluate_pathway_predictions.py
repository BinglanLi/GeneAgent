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


def extract_pathways_and_processes(file_path: Path, include_descriptions: bool = False) -> tuple[list, list, list]:
    """
    Extract reference pathway names, predicted process names, and pathway descriptions from Final_Response_GeneAgent.txt.
    
    Args:
        file_path: Path to Final_Response_GeneAgent.txt file
        include_descriptions: If True, includes pathway descriptions in the output. If False, returns a list of "None" for pathway descriptions.
    
    Returns:
        tuple: (reference_pathways, predicted_processes, pathway_descriptions)
            - reference_pathways: list of pathway names from [brackets]
            - predicted_processes: list of process names from "Process: xxx" lines
            - pathway_descriptions: list of pathway descriptions
                - If include_descriptions is True, includes pathway descriptions in the output.
                - If include_descriptions is False, returns a list of "None" for pathway descriptions.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding='utf-8') as agentfile:
        agent_text = agentfile.read()
    
    # Use existing process_text function to clean segments
    cleaned_segments = process_text(agent_text)
    reference_pathways = []
    predicted_processes = []
    pathway_descriptions = []

    for segment in cleaned_segments:
        if not segment.strip():
            continue
        
        # Extract reference pathway name from [brackets]
        bracket_match = re.search(r'\[([^\]]+)\]', segment)
        if bracket_match:
            pathway_name = bracket_match.group(1).strip()
            reference_pathways.append(pathway_name)
        else:
            reference_pathways.append("None")
        
        # Extract predicted process name from "Process: xxx" line
        lines = segment.split("\n")
        process_match = "None"
        pathway_descriptions_match = "None"
        
        for line in lines:
            line_lower = line.lower()
            if "process:" in line_lower:
                # Extract after "Process:" or "process:"
                parts = line.split(":", 1)
                if len(parts) > 1:
                    process_match = parts[1].strip()
                    # Remove trailing punctuation
                    process_match = process_match.rstrip('.,;')
                    if not include_descriptions:
                        break
    
            if include_descriptions:
                # Append any line that does not start with "[" to pathway_descriptions
                if not line.startswith("["):
                    if pathway_descriptions_match == "None":
                        pathway_descriptions_match = line.strip()
                    else:
                        pathway_descriptions_match += "\n" + line.strip()
            
        # Fallback: use "None" if no Process: found
        predicted_processes.append(process_match)
        pathway_descriptions.append(pathway_descriptions_match)
    
    return reference_pathways, predicted_processes, pathway_descriptions


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
    """
    Calculate semantic similarity using MedCPT.
    
    Args:
        reference: List of reference pathway names
        predictions: List of predicted process names
        model: MedCPT model
        tokenizer: MedCPT tokenizer
    """
    scores = []
    
    for ref, pred in tqdm(zip(reference, predictions), desc="Calculating semantic similarity", total=len(reference)):
        with torch.no_grad():
            encoded = tokenizer(
                [ref, pred],
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=512,  # Increased to accommodate descriptions
            )
            
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            score = cos_sim(embeds[0], embeds[1])
            scores.append(score.item())
    
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GeneAgent predictions against ground truth Pathway names",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV file with Pathway column'
    )
    
    parser.add_argument(
        '--llm', '-l',
        type=str,
        default='gpt-4o',
        help='LLM model name used for predictions'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: Outputs/{llm}/{dataset_name})'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file for evaluation results (default: {output_base}/evaluation_results.csv)'
    )
    
    parser.add_argument(
        '--include-descriptions',
        action='store_true',
        help='Include pathway descriptions in the evaluation (default: False)'
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
    dataset_name = input_file.stem
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        base_dir = Path(__file__).absolute().parent
        output_dir = base_dir / "Outputs" / args.llm / dataset_name
    
    full_output_dir = output_dir / "full_set"
    reduced_output_dir = output_dir / "reduced_set"
    
    full_final_file = full_output_dir / "Final_Response_GeneAgent.txt"
    reduced_final_file = reduced_output_dir / "Final_Response_GeneAgent.txt"
    
    # Extract reference pathways and predictions from Final_Response_GeneAgent.txt
    print("Extracting reference and predicted process terms from Final_Response_GeneAgent.txt files...")
    
    # Extract reference and predicted process terms from full_set
    try:
        full_reference, full_predictions, full_pathway_descriptions = extract_pathways_and_processes(full_final_file, args.include_descriptions)
        print(f"Extracted {len(full_reference)} reference process terms from full_set")
        print(f"Extracted {len(full_predictions)} predicted process terms from full_set")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Skipping full_set evaluation")
        full_reference = []
        full_predictions = []
    
    # Extract reference and predicted process terms from reduced_set
    try:
        reduced_reference, reduced_predictions, reduced_pathway_descriptions = extract_pathways_and_processes(reduced_final_file, args.include_descriptions)
        print(f"Extracted {len(reduced_reference)} reference process terms from reduced_set")
        print(f"Extracted {len(reduced_predictions)} predicted process terms from reduced_set")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping reduced_set evaluation")
        reduced_reference = []
        reduced_predictions = []
    
    # Load pathway descriptions from input CSV
    reference_pathway_descriptions = {}
    if args.include_descriptions:
        print("Loading pathway descriptions from input CSV...")
        pattern = r'\([^)]*\)'
        df_input = pd.read_csv(input_file)
        if 'Pathway' in df_input.columns and 'Pathway_Description' in df_input.columns:
            for _, row in df_input.iterrows():
                pathway_name = row.get('Pathway', '')
                pathway_name = re.sub(pattern, '', pathway_name)
                pathway_name = pathway_name.replace('/', ' ').replace(",", " ").replace('"', "").replace("-", " ").strip()
                pathway_desc = row.get('Pathway_Description', '')
                pathway_desc = pathway_desc.strip() if pd.notna(pathway_desc) else 'None'
                reference_pathway_descriptions[pathway_name] = pathway_desc
            print(f"Loaded {len(reference_pathway_descriptions)} pathway descriptions")
        else:
            print("Warning: 'Pathway' and 'Pathway_Description' columns not found in input CSV. Proceeding without descriptions.")

    # Exclude None indices from reference and predictions
    # Find None indices of reference pathway in the final response for the full set
    full_reference_none_indices = [i for i, _ in enumerate(full_reference) if _ == "None"]
    # Find None indices of reference pathway description for the full set
    if args.include_descriptions:
        full_reference_description_none_indices = [
            i for i, ref in enumerate(full_reference) 
            if reference_pathway_descriptions[ref] == "None"
        ]
        full_reference_none_indices += full_reference_description_none_indices
    # Find None indices of annotated process names for the full set
    full_predictions_none_indices = [i for i, _ in enumerate(full_predictions) if _ == "None"]
    # Find None indices of reference pathway in the final response for the reduced set
    reduced_reference_none_indices = [i for i, _ in enumerate(reduced_reference) if _ == "None"]
    # Find None indices of reference pathway description for the reduced set
    if args.include_descriptions:
        reduced_reference_description_none_indices = [
            i for i, ref in enumerate(reduced_reference) 
            if reference_pathway_descriptions[ref] == "None"
        ]
        reduced_reference_none_indices += reduced_reference_description_none_indices
    # Find None indices of annotated process names for the reduced set
    reduced_predictions_none_indices = [i for i, _ in enumerate(reduced_predictions) if _ == "None"]
    # Combine None indices for the full set
    full_none_indices = set(full_reference_none_indices + full_predictions_none_indices)
    # Combine None indices for the reduced set
    reduced_none_indices = set(reduced_reference_none_indices + reduced_predictions_none_indices)

    # Clean up references and predictions
    print(f"Excluding {len(full_none_indices)} None values from the full set")
    print(f"Excluding {len(reduced_none_indices)} None values from the reduced set")
    full_reference = [ref for i, ref in enumerate(full_reference) if i not in full_none_indices]
    full_predictions = [pred for i, pred in enumerate(full_predictions) if i not in full_none_indices]
    reduced_reference = [ref for i, ref in enumerate(reduced_reference) if i not in reduced_none_indices]
    reduced_predictions = [pred for i, pred in enumerate(reduced_predictions) if i not in reduced_none_indices]

    # Concatenate process name with description
    if args.include_descriptions:
        full_reference = [f"{ref} {reference_pathway_descriptions[ref]}" for ref in full_reference]
        full_predictions = [f"{pred} {full_pathway_descriptions[i]}" for i, pred in enumerate(full_predictions) if i not in full_none_indices]
        reduced_reference = [f"{ref} {reference_pathway_descriptions[ref]}" for ref in reduced_reference]
        reduced_predictions = [f"{pred} {reduced_pathway_descriptions[i]}" for i, pred in enumerate(reduced_predictions) if i not in reduced_none_indices]
    
    
    # Calculate ROUGE scores
    print("\nCalculating ROUGE scores...")
    metrics = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    
    results = []
    
    # Evaluate full_set predictions
    if full_predictions:
        full_rouge = calculate_rouge_scores(full_reference, full_predictions, scorer)
        for i, (ref, pred, rouge_scores) in enumerate(zip(full_reference, full_predictions, full_rouge)):
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
        reduced_rouge = calculate_rouge_scores(reduced_reference, reduced_predictions, scorer)
        for i, (ref, pred, rouge_scores) in enumerate(zip(reduced_reference, reduced_predictions, reduced_rouge)):
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
                    full_reference, full_predictions, model, tokenizer
                )
                # Add to results
                for i, score in enumerate(full_semantic):
                    idx = i  # Index in results for full_set
                    if idx < len(results) and results[idx]["prediction_type"] == "full_set":
                        results[idx]["semantic_similarity"] = score
            
            if reduced_predictions:
                print("Calculating semantic similarity for reduced_set...")
                reduced_semantic = calculate_semantic_similarity(
                    reduced_reference, reduced_predictions, model, tokenizer
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
                        print(f"  {metric} mean: {df_type[metric].mean():.4f}")

                
                if "semantic_similarity" in df_type.columns:
                    print(f"  Semantic Similarity (MedCPT) avg: {df_type['semantic_similarity'].mean():.4f}")
                    print(f"  Semantic Similarity (MedCPT) std: {df_type['semantic_similarity'].std():.4f}")
                    print(f"  Semantic Similarity (MedCPT) min: {df_type['semantic_similarity'].min():.4f}")
                    print(f"  Semantic Similarity (MedCPT) max: {df_type['semantic_similarity'].max():.4f}")
        
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
    elif args.include_descriptions:
        output_file = output_dir / "evaluation_results_descriptionsIncluded.csv"
    else:
        output_file = output_dir / "evaluation_results_nameOnly.csv"

    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

