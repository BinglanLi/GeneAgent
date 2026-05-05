#!/usr/bin/env python3
"""
Evaluate GeneAgent predictions against ground truth Pathway names.
Compares full_set and reduced_set predictions using ROUGE scores, semantic
similarity (MedCPT), and an optional LLM-as-judge score.
"""

import re
import argparse
import torch
import pandas as pd
import numpy as np

from pathlib import Path
from rouge_score import rouge_scorer
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from scipy import stats


# ---------------------------------------------------------------------------
# LLM judge prompt templates
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert in molecular biology and bioinformatics specializing in "
    "Reactome pathway analysis. You evaluate whether a predicted biological "
    "pathway name (or description) correctly identifies the same Reactome "
    "pathway as a given reference."
)

# Used when --include-descriptions is NOT set (short title comparisons).
_JUDGE_PROMPT_NAME_ONLY = """\
###Task Description:
You are a biomedical expert evaluating whether a predicted biological pathway \
name correctly identifies the same Reactome pathway as the reference name.

Reactome pathways are organized in a hierarchy (e.g., "Signal Transduction" \
contains "MAPK family signaling cascades" which contains "ERK1 and ERK2 \
cascade"). When scoring, consider:
- Exact synonyms or equivalent names score highest.
- A predicted name at a directly adjacent hierarchical level (immediate parent \
or child) that still captures the core biology scores moderately high.
- Broader or narrower terms with meaningful semantic drift score lower.
- Ignore minor formatting differences (punctuation, capitalization, Unicode \
dashes, asterisks, or other special characters).

1. Write a brief feedback assessing whether the predicted name matches the \
reference pathway, strictly based on the score rubric below.
2. After writing a feedback, write a score that is an integer between 1 and 5.
3. The output format must look as follows: \
"Feedback: (write a feedback) [RESULT] (an integer between 1 and 5)"
4. Please do not generate any other opening, closing, or explanations.

###Reference Pathway Name (Score 5):
{reference}

###Predicted Pathway Name to Evaluate:
{prediction}

###Score Rubric:
[Does the predicted pathway name correctly identify the same biological process \
as the reference?]
Score 1: The predicted name refers to a completely different or unrelated \
biological process or system.
Score 2: The predicted name is in a related biological domain but identifies a \
materially different process (e.g., a sibling pathway that does not subsume the \
reference).
Score 3: The predicted name partially overlaps with the reference — captures \
the same biological system or category but differs in the specific mechanism or \
molecular focus.
Score 4: The predicted name is a close synonym, near-equivalent label, or an \
immediately adjacent hierarchical level (direct parent or child) in Reactome \
that preserves the core biological meaning.
Score 5: The predicted name refers to the same specific biological process as \
the reference (exact match, clear synonym, or equivalent wording with no \
meaningful semantic difference).

###Feedback:\
"""

# Used when --include-descriptions IS set (title + mechanistic paragraph).
_JUDGE_PROMPT_WITH_DESCRIPTION = """\
###Task Description:
You are a biomedical expert evaluating whether a predicted biological pathway \
description correctly captures the same Reactome pathway as the reference.

Each entry consists of a pathway name followed by a mechanistic description. \
Evaluate whether the prediction identifies the same biological process with the \
same key molecular actors, mechanisms, and biological outcomes as the reference.

When scoring, consider:
- Whether the predicted title refers to the same specific Reactome pathway or \
a synonymous one.
- Whether the mechanistic description preserves the key molecules (genes, \
proteins, complexes) and their functional roles.
- Whether critical steps or relationships described in the reference are \
present, missing, or incorrect in the prediction.
- Ignore minor formatting artifacts (Unicode hyphens, asterisks, \
capitalization, markdown symbols, trailing punctuation).

1. Write a brief feedback assessing the semantic similarity between the \
prediction and reference, strictly based on the score rubric below.
2. After writing a feedback, write a score that is an integer between 1 and 5.
3. The output format must look as follows: \
"Feedback: (write a feedback) [RESULT] (an integer between 1 and 5)"
4. Please do not generate any other opening, closing, or explanations.

###Reference Pathway (Score 5):
{reference}

###Predicted Pathway to Evaluate:
{prediction}

###Score Rubric:
[Does the prediction correctly capture the same biological pathway with accurate \
mechanistic details and key molecular actors?]
Score 1: The prediction describes a completely different or unrelated biological \
process, or contains hallucinated or contradictory biology.
Score 2: The prediction addresses a related biological domain but with \
substantial semantic drift — the core mechanism is different, key molecular \
actors are wrong, or the biological outcome diverges meaningfully.
Score 3: The prediction captures the same general biological category and the \
title is acceptable, but the description is missing a key mechanism, important \
molecular actors, or a critical relationship present in the reference.
Score 4: The prediction correctly identifies the same pathway with the right \
core mechanism and key molecules; only minor omissions, additions, or wording \
differences that do not change the biological meaning.
Score 5: The prediction fully captures the same specific biological process: \
the title is semantically equivalent and the description preserves the same \
molecular mechanism, key molecular actors, and biological outcome with no \
important omissions or errors.

###Feedback:\
"""

_RESULT_PATTERN = re.compile(r'\[RESULT\]\s*([1-5])', re.IGNORECASE)
_TRAILING_DIGIT = re.compile(r'(?:score[:\s]+|result[:\s]+)?([1-5])\s*$', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

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
    Extract reference pathway names, predicted process names, and pathway
    descriptions from Final_Response_GeneAgent.txt.

    Returns:
        tuple: (reference_pathways, predicted_processes, pathway_descriptions)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding='utf-8') as agentfile:
        agent_text = agentfile.read()

    cleaned_segments = process_text(agent_text)
    reference_pathways = []
    predicted_processes = []
    pathway_descriptions = []

    for segment in cleaned_segments:
        if not segment.strip():
            continue

        bracket_match = re.search(r'\[([^\]]+)\]', segment)
        if bracket_match:
            pathway_name = bracket_match.group(1).strip()
            reference_pathways.append(pathway_name)
        else:
            reference_pathways.append("None")

        lines = segment.split("\n")
        process_match = "None"
        pathway_descriptions_match = "None"

        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith("process:"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    process_match = parts[1].strip()
                    process_match = process_match.rstrip('.,;')
                if not include_descriptions:
                    break

            if not line.startswith("["):
                if pathway_descriptions_match == "None":
                    pathway_descriptions_match = line.strip()
                else:
                    pathway_descriptions_match += " " + line.rstrip()

        predicted_processes.append(process_match)
        pathway_descriptions.append(pathway_descriptions_match)

    return reference_pathways, predicted_processes, pathway_descriptions


# ---------------------------------------------------------------------------
# ROUGE scoring
# ---------------------------------------------------------------------------

def calculate_rouge_scores(reference: list, predictions: list, scorer) -> list:
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


# ---------------------------------------------------------------------------
# MedCPT semantic similarity
# ---------------------------------------------------------------------------

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


def calculate_semantic_similarity(reference: list, predictions: list, model, tokenizer) -> list:
    """Calculate semantic similarity using MedCPT."""
    scores = []
    for ref, pred in tqdm(zip(reference, predictions), desc="Calculating semantic similarity", total=len(reference)):
        with torch.no_grad():
            encoded = tokenizer(
                [ref, pred],
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=512,
            )
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            score = cos_sim(embeds[0], embeds[1])
            scores.append(score.item())
    return scores


# ---------------------------------------------------------------------------
# LLM-as-judge scoring
# ---------------------------------------------------------------------------

def _parse_judge_response(response: str) -> tuple[int | None, str]:
    """
    Extract (score, feedback) from an LLM judge response.

    Returns (None, raw_response) if no parseable score is found.
    """
    response = response.strip()

    m = _RESULT_PATTERN.search(response)
    if m:
        score = int(m.group(1))
        feedback = response[:m.start()].strip()
        feedback = re.sub(r'^feedback:\s*', '', feedback, flags=re.IGNORECASE).strip()
        return score, feedback

    m = _TRAILING_DIGIT.search(response)
    if m:
        score = int(m.group(1))
        feedback = response[:m.start()].strip()
        feedback = re.sub(r'^feedback:\s*', '', feedback, flags=re.IGNORECASE).strip()
        return score, feedback

    return None, response


def calculate_llm_judge_scores(
    reference: list,
    predictions: list,
    judge_model: str,
    include_descriptions: bool,
) -> list[tuple[int | None, str]]:
    """
    Score each (reference, prediction) pair using an LLM judge.

    Args:
        reference: List of reference pathway names (or name + description).
        predictions: List of predicted process names (or name + description).
        judge_model: Model name passed to SimpleLLMClient.
        include_descriptions: Selects the prompt template (name-only vs.
            title + mechanistic description).

    Returns:
        List of (score, feedback) tuples in the same order as the inputs.
        score is an integer 1–5, or None if the response could not be parsed.
    """
    from llm_utils import get_llm_client

    prompt_template = (
        _JUDGE_PROMPT_WITH_DESCRIPTION if include_descriptions
        else _JUDGE_PROMPT_NAME_ONLY
    )
    client = get_llm_client(judge_model, temperature=0.0)

    results = []
    parse_failures = 0

    for ref, pred in tqdm(
        zip(reference, predictions),
        desc="LLM judge scoring",
        total=len(reference),
    ):
        prompt = prompt_template.format(reference=ref, prediction=pred)
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response_text, _ = client.chat(messages)
            score, feedback = _parse_judge_response(response_text)
            if score is None:
                parse_failures += 1
            results.append((score, feedback))
        except Exception as exc:
            print(f"\nWarning: LLM judge call failed: {exc}")
            results.append((None, f"ERROR: {exc}"))

    if parse_failures:
        print(
            f"Warning: could not parse score for {parse_failures} row(s); "
            "inspect llm_judge_feedback for raw output."
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GeneAgent predictions against ground truth Pathway names",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        nargs='+',
        help='Paths to input CSV files with Pathway column'
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
        help='Output file for evaluation results'
    )
    parser.add_argument(
        '--include-descriptions',
        action='store_true',
        help='Include pathway descriptions in the evaluation (default: False)'
    )
    parser.add_argument(
        '--skip-semantic',
        action='store_true',
        help='Skip MedCPT semantic similarity calculation'
    )
    parser.add_argument(
        '--judge-llm',
        type=str,
        default=None,
        metavar='MODEL',
        help=(
            'Run LLM-as-judge evaluation using this model (e.g. gpt-4o). '
            'Omit to skip judge scoring.'
        ),
    )

    args = parser.parse_args()

    # Resolve paths
    input_files = [Path(_).resolve() for _ in args.input]
    for input_file in input_files:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

    base_dir = Path(__file__).absolute().parent
    results_dir = base_dir / "Outputs" / args.llm
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = results_dir / input_files[0].stem

    # Accumulate across input files
    combined_full_reference = []
    combined_full_predictions = []
    combined_full_pathway_descriptions = []
    combined_reduced_reference = []
    combined_reduced_predictions = []
    combined_reduced_pathway_descriptions = []
    reference_pathway_descriptions = {}

    for input_file in input_files:
        dataset_name = input_file.stem
        print(f"Processing dataset: {dataset_name}")

        full_results_dir = results_dir / dataset_name / "full_set"
        reduced_results_dir = results_dir / dataset_name / "reduced_set"

        full_final_file = full_results_dir / "Final_Response_GeneAgent.txt"
        reduced_final_file = reduced_results_dir / "Final_Response_GeneAgent.txt"

        if args.include_descriptions:
            print(f"Loading pathway descriptions from {input_file}...")
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
                print(f"Loaded {len(reference_pathway_descriptions)} pathway descriptions so far")
            else:
                print(f"Warning: 'Pathway' and 'Pathway_Description' columns not found in {input_file}.")

        try:
            full_reference, full_predictions, full_pathway_descriptions = extract_pathways_and_processes(full_final_file, args.include_descriptions)
            print(f"Extracted {len(full_reference)} reference / {len(full_predictions)} predicted terms from full_set for {dataset_name}")
            combined_full_reference.extend(full_reference)
            combined_full_predictions.extend(full_predictions)
            combined_full_pathway_descriptions.extend(full_pathway_descriptions)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Skipping full_set evaluation for {dataset_name}")

        try:
            reduced_reference, reduced_predictions, reduced_pathway_descriptions = extract_pathways_and_processes(reduced_final_file, args.include_descriptions)
            print(f"Extracted {len(reduced_reference)} reference / {len(reduced_predictions)} predicted terms from reduced_set for {dataset_name}")
            combined_reduced_reference.extend(reduced_reference)
            combined_reduced_predictions.extend(reduced_predictions)
            combined_reduced_pathway_descriptions.extend(reduced_pathway_descriptions)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping reduced_set evaluation for {dataset_name}")

    full_reference = combined_full_reference
    full_predictions = combined_full_predictions
    full_pathway_descriptions = combined_full_pathway_descriptions
    reduced_reference = combined_reduced_reference
    reduced_predictions = combined_reduced_predictions
    reduced_pathway_descriptions = combined_reduced_pathway_descriptions

    # Filter None entries
    full_reference_none_indices = [i for i, _ in enumerate(full_reference) if _ == "None"]
    if args.include_descriptions:
        full_reference_description_none_indices = [
            i for i, ref in enumerate(full_reference)
            if reference_pathway_descriptions.get(ref) == "None"
        ]
        full_reference_none_indices += full_reference_description_none_indices
    full_predictions_none_indices = [i for i, _ in enumerate(full_predictions) if _ == "None"]

    reduced_reference_none_indices = [i for i, _ in enumerate(reduced_reference) if _ == "None"]
    if args.include_descriptions:
        reduced_reference_description_none_indices = [
            i for i, ref in enumerate(reduced_reference)
            if reference_pathway_descriptions.get(ref) == "None"
        ]
        reduced_reference_none_indices += reduced_reference_description_none_indices
    reduced_predictions_none_indices = [i for i, _ in enumerate(reduced_predictions) if _ == "None"]

    full_none_indices = set(full_reference_none_indices + full_predictions_none_indices)
    reduced_none_indices = set(reduced_reference_none_indices + reduced_predictions_none_indices)

    print(f"Excluding {len(full_none_indices)} None values from the full set")
    print(f"Excluding {len(reduced_none_indices)} None values from the reduced set")

    full_reference = [ref for i, ref in enumerate(full_reference) if i not in full_none_indices]
    full_predictions = [pred for i, pred in enumerate(full_predictions) if i not in full_none_indices]
    full_pathway_descriptions = [desc for i, desc in enumerate(full_pathway_descriptions) if i not in full_none_indices]
    reduced_reference = [ref for i, ref in enumerate(reduced_reference) if i not in reduced_none_indices]
    reduced_predictions = [pred for i, pred in enumerate(reduced_predictions) if i not in reduced_none_indices]
    reduced_pathway_descriptions = [desc for i, desc in enumerate(reduced_pathway_descriptions) if i not in reduced_none_indices]

    if args.include_descriptions:
        full_reference = [f"{ref} {reference_pathway_descriptions[ref]}" for ref in full_reference]
        full_predictions = [f"{pred} {desc}" for pred, desc in zip(full_predictions, full_pathway_descriptions)]
        reduced_reference = [f"{ref} {reference_pathway_descriptions[ref]}" for ref in reduced_reference]
        reduced_predictions = [f"{pred} {desc}" for pred, desc in zip(reduced_predictions, reduced_pathway_descriptions)]

    # -----------------------------------------------------------------------
    # ROUGE
    # -----------------------------------------------------------------------
    print("\nCalculating ROUGE scores...")
    metrics = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    results = []

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

    # -----------------------------------------------------------------------
    # MedCPT semantic similarity
    # -----------------------------------------------------------------------
    if not args.skip_semantic:
        print("\nLoading MedCPT model for semantic similarity...")
        try:
            model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
            tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

            if full_predictions:
                print("Calculating semantic similarity for full_set...")
                full_semantic = calculate_semantic_similarity(full_reference, full_predictions, model, tokenizer)
                for i, score in enumerate(full_semantic):
                    if i < len(results) and results[i]["prediction_type"] == "full_set":
                        results[i]["semantic_similarity"] = score

            if reduced_predictions:
                print("Calculating semantic similarity for reduced_set...")
                reduced_semantic = calculate_semantic_similarity(reduced_reference, reduced_predictions, model, tokenizer)
                full_count = len(full_predictions) if full_predictions else 0
                for i, score in enumerate(reduced_semantic):
                    idx = full_count + i
                    if idx < len(results) and results[idx]["prediction_type"] == "reduced_set":
                        results[idx]["semantic_similarity"] = score

        except Exception as e:
            print(f"Warning: Could not calculate semantic similarity: {e}")
            print("Continuing without semantic similarity scores...")

    # -----------------------------------------------------------------------
    # LLM-as-judge
    # -----------------------------------------------------------------------
    if args.judge_llm:
        print(f"\nRunning LLM-as-judge evaluation with model: {args.judge_llm}")
        mode_label = "with_description" if args.include_descriptions else "name_only"
        print(f"Judge mode: {mode_label}")

        if full_predictions:
            print("Scoring full_set predictions...")
            full_judge = calculate_llm_judge_scores(
                full_reference, full_predictions, args.judge_llm, args.include_descriptions
            )
            for i, (score, feedback) in enumerate(full_judge):
                if i < len(results) and results[i]["prediction_type"] == "full_set":
                    results[i]["llm_judge_score"] = score
                    results[i]["llm_judge_feedback"] = feedback

        if reduced_predictions:
            print("Scoring reduced_set predictions...")
            reduced_judge = calculate_llm_judge_scores(
                reduced_reference, reduced_predictions, args.judge_llm, args.include_descriptions
            )
            full_count = len(full_predictions) if full_predictions else 0
            for i, (score, feedback) in enumerate(reduced_judge):
                idx = full_count + i
                if idx < len(results) and results[idx]["prediction_type"] == "reduced_set":
                    results[idx]["llm_judge_score"] = score
                    results[idx]["llm_judge_feedback"] = feedback

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    if len(df_results) > 0:
        for pred_type in ["full_set", "reduced_set"]:
            df_type = df_results[df_results["prediction_type"] == pred_type]
            if len(df_type) == 0:
                continue

            print(f"\n{pred_type.upper()}:")
            print(f"  Number of predictions: {len(df_type)}")

            for metric in metrics:
                if metric in df_type.columns:
                    print(f"  {metric} mean: {df_type[metric].mean():.4f}")

            if "semantic_similarity" in df_type.columns:
                ss = df_type["semantic_similarity"].dropna()
                ss_sem = ss.std(ddof=1) / np.sqrt(len(ss))
                ss_ci = stats.t.interval(0.95, df=len(ss) - 1, loc=ss.mean(), scale=ss_sem)
                print(f"  Semantic Similarity (MedCPT) avg: {ss.mean():.4f}")
                print(f"  Semantic Similarity (MedCPT) 95% CI: ({ss_ci[0]:.4f}, {ss_ci[1]:.4f})")
                print(f"  Semantic Similarity (MedCPT) min: {ss.min():.4f}")
                print(f"  Semantic Similarity (MedCPT) max: {ss.max():.4f}")

            if "llm_judge_score" in df_type.columns:
                js = pd.to_numeric(df_type["llm_judge_score"], errors="coerce").dropna()
                if len(js) > 1:
                    js_sem = js.std(ddof=1) / np.sqrt(len(js))
                    js_ci = stats.t.interval(0.95, df=len(js) - 1, loc=js.mean(), scale=js_sem)
                    dist = "  ".join(f"{v}:{int((js == v).sum())}" for v in range(1, 6))
                    print(f"  LLM Judge Score avg: {js.mean():.4f}")
                    print(f"  LLM Judge Score 95% CI: ({js_ci[0]:.4f}, {js_ci[1]:.4f})")
                    print(f"  LLM Judge Score min: {js.min():.0f}  max: {js.max():.0f}")
                    print(f"  LLM Judge Score distribution: {dist}")

        # Comparison across conditions
        df_full = df_results[df_results["prediction_type"] == "full_set"]
        df_reduced = df_results[df_results["prediction_type"] == "reduced_set"]
        if len(df_full) > 0 and len(df_reduced) > 0:
            print("\nCOMPARISON (full_set vs reduced_set):")
            for metric in metrics:
                if metric in df_full.columns and metric in df_reduced.columns:
                    fm, rm = df_full[metric].mean(), df_reduced[metric].mean()
                    print(f"  {metric}: full={fm:.4f}  reduced={rm:.4f}  diff={fm - rm:+.4f}")

            if "semantic_similarity" in df_full.columns and "semantic_similarity" in df_reduced.columns:
                fm = df_full["semantic_similarity"].mean()
                rm = df_reduced["semantic_similarity"].mean()
                print(f"  Semantic Similarity: full={fm:.4f}  reduced={rm:.4f}  diff={fm - rm:+.4f}")

            if "llm_judge_score" in df_full.columns and "llm_judge_score" in df_reduced.columns:
                fm = pd.to_numeric(df_full["llm_judge_score"], errors="coerce").mean()
                rm = pd.to_numeric(df_reduced["llm_judge_score"], errors="coerce").mean()
                print(f"  LLM Judge Score:      full={fm:.4f}  reduced={rm:.4f}  diff={fm - rm:+.4f}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    if args.output_file:
        output_file = Path(args.output_file)
    elif args.include_descriptions:
        output_file = output_dir / "evaluation_results_descriptionsIncluded.csv"
    else:
        output_file = output_dir / "evaluation_results_nameOnly.csv"

    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
