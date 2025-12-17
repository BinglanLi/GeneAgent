#!/usr/bin/env python3
"""
Script to run GeneAgent on full_set and reduced_set columns from selected_pathways_with_gene_sets2.csv
and prepare for evaluation.
"""

import pandas as pd
import ast
import subprocess
import sys
from pathlib import Path
import argparse
import tempfile
import shutil


def parse_gene_list(gene_str):
    """Parse gene list from string representation of Python list."""
    if pd.isna(gene_str) or str(gene_str).strip() == '':
        return ''
    
    gene_str = str(gene_str).strip()
    
    try:
        # Try to evaluate as Python literal
        gene_list = ast.literal_eval(gene_str)
        if isinstance(gene_list, list):
            # Join with commas, filtering out empty strings
            genes = [str(g).strip() for g in gene_list if str(g).strip()]
            return ','.join(genes)
        else:
            return str(gene_list)
    except (ValueError, SyntaxError):
        # If parsing fails, try to clean and split
        gene_str = gene_str.strip("[]'\"")
        genes = [g.strip() for g in gene_str.split(',') if g.strip()]
        return ','.join(genes)


def prepare_csv(input_file: Path, genes_column: str, output_file: Path):
    """Prepare CSV file with ID and Genes columns for main_cascade.py."""
    df = pd.read_csv(input_file)
    
    # Remove rows with missing Pathway or gene column
    df = df.dropna(subset=['Pathway', genes_column])
    
    # Create ID column from Pathway name
    df['ID'] = df['Pathway']
    
    # Parse and convert gene lists to comma-separated strings
    df['Genes'] = df[genes_column].apply(parse_gene_list)
    
    # Select only ID and Genes columns
    result_df = df[['ID', 'Genes']].copy()
    
    # Remove rows with empty genes
    result_df = result_df[result_df['Genes'].str.strip() != '']
    
    # Save to output file
    result_df.to_csv(output_file, index=False)
    print(f"Prepared CSV with {len(result_df)} rows: {output_file}")
    
    return result_df


def run_main_cascade(input_csv: Path, llm_model: str, output_dir: Path, dataset_suffix: str):
    """Run main_cascade.py on the prepared CSV."""
    cmd = [
        sys.executable,
        "main_cascade.py",
        "--input", str(input_csv),
        "--llm", llm_model,
        "--output", str(output_dir),
        "--id-column", "ID",
        "--genes-column", "Genes"
    ]
    
    print(f"\n{'='*60}")
    print(f"Running main_cascade.py for {dataset_suffix}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"main_cascade.py failed with return code {result.returncode}")
    
    print(f"\nCompleted processing for {dataset_suffix}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run GeneAgent on full_set and reduced_set columns",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--llm', '-l',
        type=str,
        default='gpt-4o',
        help='LLM model name (default: gpt-4o)'
    )
    
    parser.add_argument(
        '--output-base', '-o',
        type=str,
        default=None,
        help='Base output directory (default: Outputs/{llm}/selected_pathways_with_gene_sets2)'
    )
    
    parser.add_argument(
        '--skip-full',
        action='store_true',
        help='Skip processing full_set (if already done)'
    )
    
    parser.add_argument(
        '--skip-reduced',
        action='store_true',
        help='Skip processing reduced_set (if already done)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of pathways to process (for testing)'
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
    
    # Create temporary directory for prepared CSVs
    temp_dir = Path(tempfile.mkdtemp(prefix="pathway_eval_"))
    
    try:
        # Read original CSV to get Pathway names for reference
        original_df = pd.read_csv(input_file)
        if args.limit:
            original_df = original_df.head(args.limit)
        
        # Prepare and run for full_set
        if not args.skip_full:
            full_csv = temp_dir / "full_set.csv"
            prepare_csv(input_file, 'full_set', full_csv)
            if args.limit:
                # Re-read and limit
                df_full = pd.read_csv(full_csv)
                df_full.head(args.limit).to_csv(full_csv, index=False)
            run_main_cascade(full_csv, args.llm, full_output_dir, "full_set")
        else:
            print("Skipping full_set processing")
        
        # Prepare and run for reduced_set
        if not args.skip_reduced:
            reduced_csv = temp_dir / "reduced_set.csv"
            prepare_csv(input_file, 'reduced_set', reduced_csv)
            if args.limit:
                # Re-read and limit
                df_reduced = pd.read_csv(reduced_csv)
                df_reduced.head(args.limit).to_csv(reduced_csv, index=False)
            run_main_cascade(reduced_csv, args.llm, reduced_output_dir, "reduced_set")
        else:
            print("Skipping reduced_set processing")
        
        # Save reference file with Pathway names for evaluation
        reference_file = base_output / "reference_pathways.csv"
        original_df[['Pathway']].to_csv(reference_file, index=False)
        print(f"\nSaved reference pathways to: {reference_file}")
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Full set results: {full_output_dir}")
        print(f"Reduced set results: {reduced_output_dir}")
        print(f"Reference pathways: {reference_file}")
        print(f"{'='*60}")
        print("\nNext step: Run evaluate_pathway_predictions.py to compare predictions against ground truth.")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

