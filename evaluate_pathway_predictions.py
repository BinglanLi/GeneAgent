#!/usr/bin/env python3
"""
Evaluate GeneAgent predictions against ground truth Pathway names.
Compares full_set, reduced_set, and noise_* predictions using ROUGE scores,
semantic similarity (MedCPT), and an optional LLM-as-judge score.
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

###Reference Pathway Name (Score 4):
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
Score 3: The predicted name is a close synonym, near-equivalent label, or an \
immediately adjacent hierarchical level (direct parent or child) in Reactome \
that preserves the core biological meaning.
Score 4: The predicted name refers to the same specific biological process as \
the reference (exact match, clear synonym, or equivalent wording with no \
meaningful semantic difference).

###Examples:
<score-4-example>
Reference title: "Assembly of the pre-replicative complex"
Prediction title: "Formation of the pre-replicative complex"
</score-4-example>

<score-3-example>
Reference title: "Assembly of the pre-replicative complex"
Prediction title: "DNA Replication"
</score-3-example>

<score-3-example>
Reference title: "Assembly of the pre-replicative complex"
Prediction title: "Assembly of the ORC complex at the origin of replication"
</score-3-example>

<score-2-example>
Reference title: "Assembly of the pre-replicative complex"
Prediction title: "Activation of the pre-replicative complex"
</score-2-example>

<score-1-example>
Reference title: "Assembly of the pre-replicative complex"
Prediction title: "DNA replication initiation"
</score-1-example>


###Feedback:\
"""

# Used when --include-descriptions IS set (title + mechanistic paragraph).
_JUDGE_PROMPT_WITH_DESCRIPTION = """\
###Task Description:
You are a biomedical expert evaluating whether a predicted biological pathway \
description correctly captures the same ground-truth Reactome pathway as the reference.

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
2. After writing a feedback, write a score that is an integer between 1 and 4.
3. The output format must look as follows: \
"Feedback: (write a feedback) [RESULT] (an integer between 1 and 5)"
4. Please do not generate any other opening, closing, or explanations.

###Reference Pathway (Score 4):
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
Score 4: The prediction fully captures the same specific biological process: \
the title is semantically equivalent and the description preserves the same \
molecular mechanism, key molecular actors, and biological outcome with no \
important omissions or errors.

###Examples:
<score-4-example>
Reference description: "Assembly of the pre-replicative complex. DNA replication pre-initiation in eukaryotic cells begins with the formation of the pre-replicative complex (pre-RC) during the late M phase and continues in the G1 phase of the mitotic cell cycle, a process also called DNA replication origin licensing. The association of initiation proteins (ORC, Cdc6, Cdt1, Mcm2-7) with the origin of replication in both S. cerevisiae and humans has been demonstrated by chromatin immunoprecipitation experiments. In S. cerevisiae, pre-replicative complexes are assembled from late M to G1. In mammalian cells as well, pre-replicative complexes are assembled from late M to G1, as shown by biochemical fractionation and immunostaining. There are significant sequence similarities among some of the proteins in the pre-replicative complex. The ORC subunits Orc1, Orc4 and Orc5 are homologous to one another and to Cdc6. The six subunits of the Mcm2-7 complex are homologous to one another. In addition, Orc1, Orc4, Orc5, Cdc6, and the Mcm2-7 subunits, are members of the AAA+ superfamily of ATPases. Since the initial identification of these pre-RC components other factors that participate in this complex have been found, including Cdt1 in human, Xenopus, S. pombe, and S. cerevisiae cells."
Prediction description: "Formation of the pre-replicative complex. DNA replication pre-initiation in eukaryotic cells begins with the formation of the pre-replicative complex (pre-RC) during the late M phase and continues in the G1 phase of the mitotic cell cycle, a process also called DNA replication origin licensing. The association of initiation proteins (ORC, Cdc6, Cdt1, Mcm2-7) with the origin of replication in both S. cerevisiae and humans has been demonstrated by chromatin immunoprecipitation experiments. In S. cerevisiae, pre-replicative complexes are assembled from late M to G1. In mammalian cells as well, pre-replicative complexes are assembled from late M to G1, as shown by biochemical fractionation and immunostaining. There are significant sequence similarities among some of the proteins in the pre-replicative complex. The ORC subunits Orc1, Orc4 and Orc5 are homologous to one another and to Cdc6. The six subunits of the Mcm2-7 complex are homologous to one another. In addition, Orc1, Orc4, Orc5, Cdc6, and the Mcm2-7 subunits, are members of the AAA+ superfamily of ATPases. Since the initial identification of these pre-RC components other factors that participate in this complex have been found, including Cdt1 in human, Xenopus, S. pombe, and S. cerevisiae cells."
</score-4-example>

<score-3-example>
Reference description: "Assembly of the pre-replicative complex. DNA replication pre-initiation in eukaryotic cells begins with the formation of the pre-replicative complex (pre-RC) during the late M phase and continues in the G1 phase of the mitotic cell cycle, a process also called DNA replication origin licensing. The association of initiation proteins (ORC, Cdc6, Cdt1, Mcm2-7) with the origin of replication in both S. cerevisiae and humans has been demonstrated by chromatin immunoprecipitation experiments. In S. cerevisiae, pre-replicative complexes are assembled from late M to G1. In mammalian cells as well, pre-replicative complexes are assembled from late M to G1, as shown by biochemical fractionation and immunostaining. There are significant sequence similarities among some of the proteins in the pre-replicative complex. The ORC subunits Orc1, Orc4 and Orc5 are homologous to one another and to Cdc6. The six subunits of the Mcm2-7 complex are homologous to one another. In addition, Orc1, Orc4, Orc5, Cdc6, and the Mcm2-7 subunits, are members of the AAA+ superfamily of ATPases. Since the initial identification of these pre-RC components other factors that participate in this complex have been found, including Cdt1 in human, Xenopus, S. pombe, and S. cerevisiae cells."
Prediction description: "DNA Replication. Studies in the past decade have suggested that the basic mechanism of DNA replication initiation is conserved in all kingdoms of life. Initiation in unicellular eukaryotes, in particular Saccharomyces cerevisiae (budding yeast), is well understood, and has served as a model for studies of DNA replication initiation in multicellular eukaryotes, including humans. In general terms, the first step of initiation is the binding of the replication initiator to the origin of replication. The replicative helicase is then assembled onto the origin, usually by a helicase assembly factor. Either shortly before or shortly after helicase assembly, some local unwinding of the origin of replication occurs in a region rich in adenine and thymine bases (often termed a DNA unwinding element, DUE). The unwound region provides the substrate for primer synthesis and initiation of DNA replication. The best-defined eukaryotic origins are those of S. cerevisiae, which have well-conserved sequence elements for initiator binding, DNA unwinding and binding of accessory proteins. In multicellular eukaryotes, unlike S. cerevisiae, these loci appear not to be defined by the presence of a DNA sequence motif. Indeed, choice of replication origins in a multicellular eukaryote may vary with developmental stage and tissue type. In cell-free models of metazoan DNA replication, such as the one provided by Xenopus egg extracts, there are only limited DNA sequence specificity requirements for replication initiation (Kelly & Brown 2000; Bell & Dutta 2002; Marahrens & Stillman 1992; Cimbora & Groudine 2001; Mahbubani et al 1992, Hyrien & Mechali 1993)."
</score-3-example>

<score-3-example>
Reference description: "Assembly of the pre-replicative complex. DNA replication pre-initiation in eukaryotic cells begins with the formation of the pre-replicative complex (pre-RC) during the late M phase and continues in the G1 phase of the mitotic cell cycle, a process also called DNA replication origin licensing. The association of initiation proteins (ORC, Cdc6, Cdt1, Mcm2-7) with the origin of replication in both S. cerevisiae and humans has been demonstrated by chromatin immunoprecipitation experiments. In S. cerevisiae, pre-replicative complexes are assembled from late M to G1. In mammalian cells as well, pre-replicative complexes are assembled from late M to G1, as shown by biochemical fractionation and immunostaining. There are significant sequence similarities among some of the proteins in the pre-replicative complex. The ORC subunits Orc1, Orc4 and Orc5 are homologous to one another and to Cdc6. The six subunits of the Mcm2-7 complex are homologous to one another. In addition, Orc1, Orc4, Orc5, Cdc6, and the Mcm2-7 subunits, are members of the AAA+ superfamily of ATPases. Since the initial identification of these pre-RC components other factors that participate in this complex have been found, including Cdt1 in human, Xenopus, S. pombe, and S. cerevisiae cells."
Prediction description: "Assembly of the ORC complex at the origin of replication. Human ORC1 can associate with DNA origin of replication sites independently of other origin of replication complex (ORC) subunits (Hoshina et al. 2013; Eladl et al. 2021). ORC1 localizes to condensed chromosomes during early mitosis (M phase) and serves as a nucleating center for the assembly of the ORC and, subsequently, the pre-replication complex. ORC1 remains associated with late replication origins throughout late G1. Upon S phase entry, ORC1 undergoes ubiquitin-mediated degradation, leading to dissociation of the ORC from chromatin (Kara et al. 2015). Most human replication origins contain guanine (G)-rich sequences which may form G-quadruplex (G4) structures (Besnard et al. 2012) and these G4 structures may mediate the recognition of replication origins by ORC1 (Hoshina et al. 2013; Eladl et al. 2021). Besides binding to nucleosome-free replication origin DNA, ORC1 interacts with neighboring nucleosomes (Hizume et al. 2013), in particular with nucleosomes containing histone H4 dimethylated at lysine 21 (H4K20me2 mark), which is enriched at replication origins. Binding of ORC1 to H4K20me2 facilitates ORC1 binding to replication origins and ORC chromatin loading (Kuo et al. 2012, Zhang et al. 2015). ORC1 binding sites are universally associated with transcription start sites (TSSs) of coding and non-coding RNAs. Replication origins associated with moderate to high transcription level TSSs (belonging to coding RNAs) fire in early S phase, while those associated with low transcription level TSSs (belonging to non-coding RNAs) fire throughout the S phase (Dellino et al. 2013). ORC2 forms a heterodimer with ORC3, which is a prerequisite for the association of ORC5 and, subsequently, ORC4 (Ranjan and Gossen 2006; Siddiqui and Stillman 2007). ORC1 binds to the ORC(2-5) complex in the nucleus to form a stable ORC(1-5) complex (Radichev et al. 2006; Ghosh et al. 2011). ORC1 is necessary for the association of the ORC(2-5) complex to chromatin (Radichev et al. 2006). The ORC(2-5) complex exhibits a tightly autoinhibited conformation, with the winged-helix domain (WHD) of ORC2 completely blocking the central DNA-binding channel. Binding of ORC1 remodels the WHD of ORC2, moving it away from the central channel and partially relieving the autoinhibition (Cheng et al. 2020, Jaremko et al. 2020). ORC6 associates with the ORC(1-5) complex to form the ORC(1-6) complex (Ghosh et al. 2011). The association of ORC6 with the ORC(1-5) complex is weak and it frequently does not co-immunoprecipitate with the other ORC(1-5) subunits. ORC4 is the only ORC(1-5) subunit that was shown to directly bind to ORC6 (Radichev et al. 2006). Some ORC6 mutations reported in Meier-Gorlin syndrome were shown to interfere with ORC6 incorporation into the ORC (Balasov et al. 2015)."
</score-3-example>

<score-2-example>
Reference description: "Assembly of the pre-replicative complex. DNA replication pre-initiation in eukaryotic cells begins with the formation of the pre-replicative complex (pre-RC) during the late M phase and continues in the G1 phase of the mitotic cell cycle, a process also called DNA replication origin licensing. The association of initiation proteins (ORC, Cdc6, Cdt1, Mcm2-7) with the origin of replication in both S. cerevisiae and humans has been demonstrated by chromatin immunoprecipitation experiments. In S. cerevisiae, pre-replicative complexes are assembled from late M to G1. In mammalian cells as well, pre-replicative complexes are assembled from late M to G1, as shown by biochemical fractionation and immunostaining. There are significant sequence similarities among some of the proteins in the pre-replicative complex. The ORC subunits Orc1, Orc4 and Orc5 are homologous to one another and to Cdc6. The six subunits of the Mcm2-7 complex are homologous to one another. In addition, Orc1, Orc4, Orc5, Cdc6, and the Mcm2-7 subunits, are members of the AAA+ superfamily of ATPases. Since the initial identification of these pre-RC components other factors that participate in this complex have been found, including Cdt1 in human, Xenopus, S. pombe, and S. cerevisiae cells."
Prediction description: "Activation of the pre-replicative complex. In S. cerevisiae, two ORC subunits, Orc1 and Orc5, both bind ATP, and Orc1 in addition has ATPase activity. Both ATP binding and ATP hydrolysis appear to be essential functions in vivo. ATP binding by Orc1 is unaffected by the association of ORC with origin DNA (ARS) sequences, but ATP hydrolysis is ARS-dependent, being suppressed by associated double-stranded DNA and stimulated by associated single-stranded DNA. These data are consistent with the hypothesis that ORC functions as an ATPase switch, hydrolyzing bound ATP and changing state as DNA unwinds at the origin immediately before replication. It is attractive to speculate that ORC likewise functions as a switch as human pre-replicative complexes are activated, but human Orc proteins are not well enough characterized to allow the model to be critically tested. mRNAs encoding human orthologs of all six Orc proteins have been cloned, and ATP-binding amino acid sequence motifs have been identified in Orc1, Orc4, and Orc5. Interactions among proteins expressed from the cloned genes have been characterized, but the ATP-binding and hydrolyzing properties of these proteins and complexes of them have not been determined."
</score-2-example>

<score-1-example>
Reference description: "Assembly of the pre-replicative complex. DNA replication pre-initiation in eukaryotic cells begins with the formation of the pre-replicative complex (pre-RC) during the late M phase and continues in the G1 phase of the mitotic cell cycle, a process also called DNA replication origin licensing. The association of initiation proteins (ORC, Cdc6, Cdt1, Mcm2-7) with the origin of replication in both S. cerevisiae and humans has been demonstrated by chromatin immunoprecipitation experiments. In S. cerevisiae, pre-replicative complexes are assembled from late M to G1. In mammalian cells as well, pre-replicative complexes are assembled from late M to G1, as shown by biochemical fractionation and immunostaining. There are significant sequence similarities among some of the proteins in the pre-replicative complex. The ORC subunits Orc1, Orc4 and Orc5 are homologous to one another and to Cdc6. The six subunits of the Mcm2-7 complex are homologous to one another. In addition, Orc1, Orc4, Orc5, Cdc6, and the Mcm2-7 subunits, are members of the AAA+ superfamily of ATPases. Since the initial identification of these pre-RC components other factors that participate in this complex have been found, including Cdt1 in human, Xenopus, S. pombe, and S. cerevisiae cells."
Prediction description: "DNA replication initiation. DNA polymerases are not capable of de novo DNA synthesis and require synthesis of a primer, usually by a DNA-dependent RNA polymerase (primase) to begin DNA synthesis. In eukaryotic cells, the primer is synthesized by DNA polymerase alpha:primase. First, the DNA primase portion of this complex synthesizes approximately 6-10 nucleotides of RNA primer and then the DNA polymerase portion synthesizes an additional 20 nucleotides of DNA (Frick & Richardson 2002; Wang et al 1984)."
</score-1-example>

###Feedback:\
"""

_RESULT_PATTERN = re.compile(r'\[RESULT\]\s*([1-5])', re.IGNORECASE)
_TRAILING_DIGIT = re.compile(r'(?:score[:\s]+|result[:\s]+)?([1-5])\s*$', re.IGNORECASE)


def detect_gene_set_columns(df: pd.DataFrame) -> list:
    """Return gene set columns in order: full_set, reduced_set, then noise_* sorted by level."""
    cols = [c for c in ['full_set', 'reduced_set'] if c in df.columns]
    noise_cols = sorted(
        (c for c in df.columns if c.startswith('noise_')),
        key=lambda c: int(c.split('_')[1])
    )
    return cols + noise_cols


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
                    pathway_descriptions_match = line.strip() + '. '
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
        feedback = re.sub(r'^feedback:\s*', '', feedback, flags=re.IGNORECASE).replace('\n', '').strip()
        return score, feedback

    m = _TRAILING_DIGIT.search(response)
    if m:
        score = int(m.group(1))
        feedback = response[:m.start()].strip()
        feedback = re.sub(r'^feedback:\s*', '', feedback, flags=re.IGNORECASE).replace('\n', '').strip()
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

    # Accumulate data per gene set column across all input files
    # sets_data[col] = {'reference': [...], 'predictions': [...], 'descriptions': [...]}
    sets_data = {}
    reference_pathway_descriptions = {}

    for input_file in input_files:
        dataset_name = input_file.stem
        print(f"Processing dataset: {dataset_name}")

        df_input = pd.read_csv(input_file)
        gene_set_cols = detect_gene_set_columns(df_input)

        if args.include_descriptions:
            print(f"Loading pathway descriptions from {input_file}...")
            pattern = r'\([^)]*\)'
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

        for col in gene_set_cols:
            col_final_file = results_dir / dataset_name / col / "Final_Response_GeneAgent.txt"
            try:
                ref, pred, desc = extract_pathways_and_processes(col_final_file, args.include_descriptions)
                print(f"Extracted {len(ref)} reference / {len(pred)} predicted terms from {col} for {dataset_name}")
                if col not in sets_data:
                    sets_data[col] = {'reference': [], 'predictions': [], 'descriptions': []}
                sets_data[col]['reference'].extend(ref)
                sets_data[col]['predictions'].extend(pred)
                sets_data[col]['descriptions'].extend(desc)
            except FileNotFoundError as e:
                print(f"Warning: {e}\nSkipping {col} evaluation for {dataset_name}")

    # -----------------------------------------------------------------------
    # Filter None entries per column
    # -----------------------------------------------------------------------
    for col in list(sets_data.keys()):
        data = sets_data[col]
        ref_none = {i for i, r in enumerate(data['reference']) if r == "None"}
        if args.include_descriptions:
            ref_none |= {i for i, r in enumerate(data['reference'])
                         if reference_pathway_descriptions.get(r) == "None"}
        pred_none = {i for i, p in enumerate(data['predictions']) if p == "None"}
        none_idx = ref_none | pred_none
        print(f"Excluding {len(none_idx)} None values from {col}")
        sets_data[col]['reference'] = [r for i, r in enumerate(data['reference']) if i not in none_idx]
        sets_data[col]['predictions'] = [p for i, p in enumerate(data['predictions']) if i not in none_idx]
        sets_data[col]['descriptions'] = [d for i, d in enumerate(data['descriptions']) if i not in none_idx]

    if args.include_descriptions:
        for col, data in sets_data.items():
            sets_data[col]['reference'] = [
                f"{r} {reference_pathway_descriptions[r]}" for r in data['reference']
            ]
            sets_data[col]['predictions'] = [
                f"{p} {d}" for p, d in zip(data['predictions'], data['descriptions'])
            ]

    # -----------------------------------------------------------------------
    # ROUGE
    # -----------------------------------------------------------------------
    print("\nCalculating ROUGE scores...")
    metrics = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    results = []
    col_result_ranges = {}  # col -> (start_idx, end_idx) in results list

    for col, data in sets_data.items():
        if not data['predictions']:
            continue
        start_idx = len(results)
        rouge_scores_list = calculate_rouge_scores(data['reference'], data['predictions'], scorer)
        for i, (ref, pred, rs) in enumerate(zip(data['reference'], data['predictions'], rouge_scores_list)):
            result = {
                "pathway_id": i,
                "reference": ref,
                "prediction_type": col,
                "prediction": pred,
            }
            for metric in metrics:
                result[metric] = rs[metric]
            results.append(result)
        col_result_ranges[col] = (start_idx, len(results))

    # -----------------------------------------------------------------------
    # MedCPT semantic similarity
    # -----------------------------------------------------------------------
    if not args.skip_semantic:
        print("\nLoading MedCPT model for semantic similarity...")
        try:
            model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
            tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

            for col, data in sets_data.items():
                if not data['predictions'] or col not in col_result_ranges:
                    continue
                print(f"Calculating semantic similarity for {col}...")
                scores = calculate_semantic_similarity(data['reference'], data['predictions'], model, tokenizer)
                start, _ = col_result_ranges[col]
                for i, score in enumerate(scores):
                    results[start + i]["semantic_similarity"] = score

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

        for col, data in sets_data.items():
            if not data['predictions'] or col not in col_result_ranges:
                continue
            print(f"Scoring {col} predictions...")
            judge_results = calculate_llm_judge_scores(
                data['reference'], data['predictions'], args.judge_llm, args.include_descriptions
            )
            start, _ = col_result_ranges[col]
            for i, (score, feedback) in enumerate(judge_results):
                results[start + i]["llm_judge_score"] = score
                results[start + i]["llm_judge_feedback"] = feedback

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    col_order = [c for c in ['full_set', 'reduced_set'] if c in sets_data]
    col_order += sorted(
        (c for c in sets_data if c not in {'full_set', 'reduced_set'}),
        key=lambda c: int(c.split('_')[1]) if c.startswith('noise_') else 0
    )

    if len(df_results) > 0:
        for col in col_order:
            df_col = df_results[df_results["prediction_type"] == col]
            if len(df_col) == 0:
                continue

            print(f"\n{col.upper()}:")
            print(f"  Number of predictions: {len(df_col)}")

            for metric in metrics:
                if metric in df_col.columns:
                    print(f"  {metric} mean: {df_col[metric].mean():.4f}")

            if "semantic_similarity" in df_col.columns:
                ss = df_col["semantic_similarity"].dropna()
                ss_sem = ss.std(ddof=1) / np.sqrt(len(ss))
                ss_ci = stats.t.interval(0.95, df=len(ss) - 1, loc=ss.mean(), scale=ss_sem)
                print(f"  Semantic Similarity (MedCPT) avg: {ss.mean():.4f}")
                print(f"  Semantic Similarity (MedCPT) 95% CI: ({ss_ci[0]:.4f}, {ss_ci[1]:.4f})")
                print(f"  Semantic Similarity (MedCPT) min: {ss.min():.4f}")
                print(f"  Semantic Similarity (MedCPT) max: {ss.max():.4f}")

            if "llm_judge_score" in df_col.columns:
                js = pd.to_numeric(df_col["llm_judge_score"], errors="coerce").dropna()
                if len(js) > 1:
                    js_sem = js.std(ddof=1) / np.sqrt(len(js))
                    js_ci = stats.t.interval(0.95, df=len(js) - 1, loc=js.mean(), scale=js_sem)
                    dist = "  ".join(f"{v}:{int((js == v).sum())}" for v in range(1, 6))
                    print(f"  LLM Judge Score avg: {js.mean():.4f}")
                    print(f"  LLM Judge Score 95% CI: ({js_ci[0]:.4f}, {js_ci[1]:.4f})")
                    print(f"  LLM Judge Score min: {js.min():.0f}  max: {js.max():.0f}")
                    print(f"  LLM Judge Score distribution: {dist}")

        # Comparison: each non-full_set column vs full_set
        df_full = df_results[df_results["prediction_type"] == "full_set"]
        if len(df_full) > 0:
            for col in [c for c in col_order if c != "full_set"]:
                df_col = df_results[df_results["prediction_type"] == col]
                if len(df_col) == 0:
                    continue
                print(f"\nCOMPARISON (full_set vs {col}):")
                for metric in metrics:
                    if metric in df_full.columns and metric in df_col.columns:
                        fm, cm = df_full[metric].mean(), df_col[metric].mean()
                        print(f"  {metric}: full={fm:.4f}  {col}={cm:.4f}  diff={fm - cm:+.4f}")
                if "semantic_similarity" in df_full.columns and "semantic_similarity" in df_col.columns:
                    fm = df_full["semantic_similarity"].mean()
                    cm = df_col["semantic_similarity"].mean()
                    print(f"  Semantic Similarity: full={fm:.4f}  {col}={cm:.4f}  diff={fm - cm:+.4f}")
                if "llm_judge_score" in df_full.columns and "llm_judge_score" in df_col.columns:
                    fm = pd.to_numeric(df_full["llm_judge_score"], errors="coerce").mean()
                    cm = pd.to_numeric(df_col["llm_judge_score"], errors="coerce").mean()
                    print(f"  LLM Judge Score:      full={fm:.4f}  {col}={cm:.4f}  diff={fm - cm:+.4f}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    if args.output_file:
        output_file = Path(args.output_file)
    elif args.include_descriptions:
        output_file = output_dir / "evaluation_results_analyticNarratives.csv"
    else:
        output_file = output_dir / "evaluation_results_processNames.csv"

    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
