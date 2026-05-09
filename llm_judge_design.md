# LLM-as-Judge Evaluation of Biological Pathway Predictions

## Motivation

Standard lexical metrics (ROUGE-1/2/L) and embedding-based cosine similarity suffer from systematic inflation when applied to biomedical pathway labels. Short synonyms such as "amino acid transport" and "transmembrane amino acid transport" score poorly under ROUGE despite being semantically equivalent, while superficially similar terms from different biological domains (e.g., "DNA replication" and "DNA repair synthesis") can inflate cosine similarity. We therefore introduce an LLM-as-judge evaluation layer calibrated to the semantic norms of a biomedical researcher.

---

## 2.1 Input Data

Two evaluation conditions are assessed, corresponding to two output formats produced by GeneAgent:

- **Name-only** (`evaluation_results_nameOnly.csv`): the agent's prediction is a short pathway title (e.g., *"Transmembrane amino acid transport"*). The reference is the canonical Reactome pathway name.
- **Description-included** (`evaluation_results_descriptionsIncluded.csv`): the prediction is a title plus a mechanistic paragraph listing key genes and molecular relationships. The reference is the Reactome pathway name concatenated with its curated description.

Both conditions are evaluated across two gene-set completeness regimes: `full_set` (complete gene list) and `reduced_set` (noise-perturbed gene list).

---

## 2.2 Judge Prompt Design

A single LLM judge is queried for each (reference, prediction) pair. The prompt follows the Prometheus rubric format — instruction, reference, response, and a scored rubric — with two mode-specific variants:

**Name-only prompt** instructs the judge to assess whether the predicted title refers to the same Reactome pathway as the reference, explicitly accounting for Reactome's hierarchical structure. A predicted name that is an immediate parent or child in the hierarchy (e.g., *"Coagulation"* for *"Removal of aminoterminal propeptides from gamma-carboxylated proteins"*) is penalized differently from a semantically equivalent synonym. The judge is also instructed to ignore formatting artifacts common in LLM outputs (Unicode dashes, asterisks, capitalization).

**Description-included prompt** extends the evaluation to mechanistic content. The judge assesses (a) whether the predicted title is semantically equivalent to the reference, and (b) whether the predicted description preserves the same molecular mechanism, key gene products, and biological outcome. Hallucinated gene names or contradictory mechanistic claims are penalized at the lowest score level.

---

## 2.3 Scoring Rubric

Both modes use a 4-point integer scale. The discriminating criteria differ by mode:

| Score | Name-only | Description-included |
|-------|-----------|----------------------|
| 4 | Exact match or unambiguous synonym; same specific Reactome pathway | Title equivalent + description preserves mechanism, key molecules, and outcome with no meaningful omissions |
| 3 | Closely synonymous or immediately adjacent hierarchical level (direct parent/child) | Correct pathway; description mostly accurate with only minor omissions or wording differences |
| 2 | Related biological domain, but a materially distinct process (sibling pathway or different endpoint) | Substantial semantic drift in mechanism or molecular actors |
| 1 | Unrelated or incorrect biological process | Wrong pathway, hallucinated biology, or contradictory mechanism |

The critical design decision is that **hierarchical proximity is not equivalence**: a prediction of *"Coagulation"* for *"Removal of aminoterminal propeptides from gamma-carboxylated proteins"* scores 2 (related domain, different process), not 3–4, because the predicted term does not subsume the specific enzymatic cleavage step being referenced.

---

## 2.4 Implementation

The evaluation script (`evaluate_with_llm_judge.py`) reads an existing evaluation CSV, selects the appropriate prompt template based on mode (auto-detected from the filename or supplied via flag), and calls the judge LLM for each row via the project's existing `SimpleLLMClient` abstraction. The judge's response is parsed for the `[RESULT] N` pattern with a regex fallback for trailing-digit formats. Rows where no score can be parsed are flagged rather than silently dropped.

Results are appended as two new columns — `llm_judge_score` (integer 1–4) and `llm_judge_feedback` (the judge's rationale) — to the input CSV, enabling direct comparison with existing ROUGE and MedCPT cosine similarity scores within a single file. A `--resume` flag allows interrupted runs to be continued without re-scoring already-evaluated rows.

Summary statistics (mean, 95% CI, score distribution) are reported separately for `full_set` and `reduced_set`, and the delta between conditions is reported to quantify the effect of gene-set noise on prediction quality.

---

## Key Design Decisions

1. **Temperature = 0** for the judge — reduces variance at the cost of no uncertainty quantification. An alternative is to run N=3 samples and report judge agreement as a reliability metric.
2. **Single judge model** — judge choice will affect absolute scores. Ideally the rubric is validated against a small human-annotated gold set before running at scale.
3. **Hierarchical scoring** — the rubric currently does not distinguish between *one level up* (Score 4) and *two levels up* (Score 3); for datasets with deep Reactome hierarchies this may need refinement.
4. **Name-only vs. description-included are scored independently** — scores are not directly comparable across modes because they measure different things (title equivalence vs. mechanistic fidelity).
