#!/usr/bin/env python3
"""
Script to run GeneAgent on gene set columns (full_set, reduced_set, noise_*) from
an AlzKB pathways CSV and prepare for evaluation.
"""

import pandas as pd
import ast
import subprocess
import sys
from pathlib import Path
import argparse
import tempfile
import shutil
import signal


def detect_gene_set_columns(df: pd.DataFrame) -> list:
    """Return gene set columns in order: full_set, reduced_set, then noise_* sorted by level."""
    cols = [c for c in ['full_set', 'reduced_set'] if c in df.columns]
    noise_cols = sorted(
        (c for c in df.columns if c.startswith('noise_')),
        key=lambda c: int(c.split('_')[1])
    )
    return cols + noise_cols


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
    """
    Run main_cascade.py as a subprocess with proper cleanup.
    The subprocess approach avoids import issues and ensures clean process termination.
    """
    cmd = [
        sys.executable,
        "-u",  # Unbuffered output for subprocess
        "main_cascade.py",
        "--input", str(input_csv),
        "--llm", llm_model,
        "--output", str(output_dir),
        "--id-column", "ID",
        "--genes-column", "Genes",
        "--clear-output",
    ]

    print(f"\n{'='*60}")
    print(f"Running main_cascade for {dataset_suffix}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        # Run with a very long timeout (24 hours) to detect hangs
        # The explicit sys.exit(0) in main_cascade.py should ensure proper termination
        result = subprocess.run(
            cmd,
            capture_output=False,
            timeout=86400,  # 24 hours
            # Ensure subprocess gets killed properly on parent exit
            preexec_fn=None if sys.platform == "win32" else lambda: signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        )

        if result.returncode != 0:
            raise RuntimeError(f"main_cascade failed with exit code {result.returncode}")

        print(f"\nCompleted processing for {dataset_suffix}\n")

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"main_cascade timed out after 4 hours for {dataset_suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GeneAgent on gene set columns (full_set, reduced_set, noise_*)",
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
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: Outputs/{llm}/{dataset_name})'
    )

    parser.add_argument(
        '--skip-columns',
        nargs='*',
        default=[],
        metavar='COL',
        help='Column names to skip (e.g. --skip-columns full_set noise_40)'
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

    # Set up output directory
    dataset_name = input_file.stem
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        base_dir = Path(__file__).absolute().parent
        output_dir = base_dir / "Outputs" / args.llm / dataset_name

    # Create temporary directory for prepared CSVs
    temp_dir = Path(tempfile.mkdtemp(prefix="pathway_eval_"))

    try:
        original_df = pd.read_csv(input_file)
        if args.limit:
            original_df = original_df.head(args.limit)

        columns = detect_gene_set_columns(original_df)
        skip = set(args.skip_columns or [])
        columns = [c for c in columns if c not in skip]

        print(f"Gene set columns to process: {columns}")

        for col in columns:
            col_csv = temp_dir / f"{col}.csv"
            prepare_csv(input_file, col, col_csv)
            if args.limit:
                df_col = pd.read_csv(col_csv)
                df_col.head(args.limit).to_csv(col_csv, index=False)
            run_main_cascade(col_csv, args.llm, output_dir / col, col)

        # Save reference file with Pathway names for evaluation
        reference_file = output_dir / "reference_pathways.csv"
        original_df[['Pathway']].to_csv(reference_file, index=False)
        print(f"\nSaved reference pathways to: {reference_file}")

        print(f"\n{'='*60}")
        print("Processing complete!")
        for col in columns:
            print(f"  {col} results: {output_dir / col}")
        print(f"Reference pathways: {reference_file}")
        print(f"{'='*60}")
        print("\nNext step: Run evaluate_pathway_predictions.py to compare predictions against ground truth.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

