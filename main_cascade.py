import json
import re
import sys
import atexit
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Use LLM utility module (refactored to use BaseAgent directly)
from llm_utils import get_llm_client
from costs import record_chat_completion_cost

load_dotenv()

from worker import AgentPhD


## baseline 
system = "You are an efficient and insightful assistant to a molecular biologist."

baseline = lambda genes: f"""
Write a critical analysis of the biological processes performed by this system of interacting proteins.
Propose a brief name for the most prominent biological process performed by the system. 
Put the name at the top of the analysis as "Process: <name>".
Be concise, do not use unnecessary words.
Be textual, do not use any format symbols such as "*", "-" or other tokens.
Be specific, avoid overly general statements such as "the proteins are involved in various cellular processes".
Be factual, do not editorialize.
For each important point, describe your reasoning and supporting information.
For each biological function name, show the corresponding gene names.
Here is the gene set: {genes}
"""

system_verify = "You are a helpful and objective fact-checker to verify the summary of gene set."
topic = lambda genes, process: f"""
Here is the original process name for the gene set {genes}:\n{process}
However, the process name might be false. Please generate decontextualized claims for the process name that need to be verified.
Only Return a list type that contain all generated claim strings, for example, ["claim_1", "claim_2"]
"""
topic_instruction = """
Only generate claims with affirmative sentence for the entire gene set.
The gene set should only be separated by comma, e.g., "a,b,c".
Don't generate claims for the single gene or incomplete gene set.
Don't generate hypotheis claims over the previous analysis.
Please replace the statement like 'these genes', 'this system' with the core genes in the given gene set.
"""
# Please replace the statement like 'these genes', 'this system' with the entire gene set.

analysis = lambda summ: f"""
Here is the summary of the given gene set: \n{summ}
However, the gene analysis in the summary might not support the updated process name. 
Please generate several decontextualized claims for the analytical narratives that need to be verified.
Only Return a list type that contain all generated claim strings, for example, ["claim_1", "claim_2"]
"""
analysis_instruction = """
Generate claims for genes and their biological functions around the updated process name.
Don't generate claims for the entire gene set or 'this system'.
Don't generate unworthy claims such as the summarization and reasoning over the previous analysis. 
Claims must contain the gene names and their biological process functions.
"""

modification = lambda verification_topic: f"""
I have finished the verification for process name. Here is the verification report:\n{verification_topic}
You should only consider the successfully verified claims.
If claims are supported, you should retain the original process name and only can make a minor grammar revision. 
if claims are partially supported, you should discard the unsupported part.
If claims are refuted, you must replace the original process name with the most significant (i.e., top-1) biological function term summarized from the verification report.
Meanwhile, revise the original summaries using the verified (or updated) process name. Do not use sentence like "There are no direct evidence to..."
"""

modification_instruction = """
Put the updated process name at the top of the analysis as "Process: <name>".
Be concise, do not use unnecessary words.
Be textual, do not use any format symbols such as "*", "-" or other tokens. All modified sentence should encoded into utf-8.
Be specific, avoid overly general statements such as "the proteins are involved in various cellular processes".
Be factual, do not editorialize.
You must retain the gene names of each updated biological functions in the new summary.
"""

summarization = lambda verification_analysis: f"""
I have finished the verification for the revised summary. Here is the verification report:\n{verification_analysis}
Please modify the summary according to the verification report again.
"""

summarization_instruction = """ 
If the analytical narratives of genes can't directly support or related to the updated process name, you must propose a new brief biological process name from the analytical texts. 
Otherwise, you must retain the updated process name and only can make a grammar revision.
IF the claim is supported, you must complement the narratives by using the standard evidence of gene set functions (or gene summaries) in the verification report but don't change the updated process name. 
IF the claim is not supported, do not mention any statement like "... was not directly confirmed by..."
Be concise, do not use unnecessary format like **, only return the concise texts.
"""

reposits = [
    "get_complex_for_gene_set",
    "get_disease_for_single_gene",
    "get_domain_for_single_gene",
    "get_enrichment_for_gene_set",
    "get_pathway_for_gene_set",
    "get_interactions_for_gene_set",
    "get_gene_summary_for_single_gene",
    "get_pubmed_articles"
]

# Note: agentphd is now initialized inside GeneAgent() to avoid hanging on module import


def _track_usage_and_cost(usage_metrics, llm_model, tag, total_prompt_tokens, total_completion_tokens, total_cost):
    """
    Helper function to track usage from BaseAgent's usage_metrics and calculate costs.
    
    Args:
        usage_metrics: UsageMetrics object from BaseAgent
        llm_model: Model name
        tag: Tag for cost tracking
        total_prompt_tokens, total_completion_tokens, total_cost: Running totals
    
    Returns:
        Tuple of (updated_total_prompt_tokens, updated_total_completion_tokens, updated_total_cost, cost_info_dict)
    """
    input_tokens = usage_metrics.input_tokens if usage_metrics and usage_metrics.input_tokens else 0
    output_tokens = usage_metrics.output_tokens if usage_metrics and usage_metrics.output_tokens else 0
    
    # Convert UsageMetrics to dict format
    usage_dict = {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': usage_metrics.total_tokens if usage_metrics and usage_metrics.total_tokens else (input_tokens + output_tokens)
    }
    
    # Record costs directly with usage_dict
    cost_info = record_chat_completion_cost(model=llm_model, tag=tag, usage_dict=usage_dict)
    
    total_prompt_tokens += cost_info["prompt_tokens"]
    total_completion_tokens += cost_info["completion_tokens"]
    total_cost += cost_info["total_cost"]
    
    return total_prompt_tokens, total_completion_tokens, total_cost, cost_info


def extract_json_list(content: str) -> list:
    """
    Extract a list from LLM response, handling various formats.
    
    Handles:
    - Pure JSON: ["item1", "item2"]
    - Markdown code blocks: ```json ["item1"] ```
    - Bulleted lists: * item1\n* item2 or - item1\n- item2
    - Numbered lists: 1. item1\n2. item2
    - Text with embedded JSON
    """
    # Try to parse as-is first (for well-behaved models like GPT-4)
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
        # If it's a dict or other type, wrap in list
        return [result] if result else []
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
    if code_block_match:
        try:
            result = json.loads(code_block_match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON array in the content (not in code blocks)
    json_array_match = re.search(r'\[(?:[^\[\]]|"[^"]*")*\]', content, re.DOTALL)
    if json_array_match:
        try:
            result = json.loads(json_array_match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    
    # Handle bulleted/numbered lists (common with local models)
    # Pattern: lines starting with *, -, •, or numbers like "1.", "2."
    lines = content.split('\n')
    list_items = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with common bullet/number patterns
        # Patterns: "* item", "- item", "• item", "1. item", "2) item"
        match = re.match(r'^[\*\-•+][\s]+(.+)$', line) or \
                re.match(r'^\d+[\.\)]\s+(.+)$', line)
        
        if match:
            claim = match.group(1).strip()
            if len(claim) > 5:  # Reasonable claim length
                list_items.append(claim)
    
    if list_items:
        return list_items
    
    # If we found no list items but have content, try extracting all non-empty lines
    # that look like claims (longer than a header/intro line)
    potential_claims = []
    for line in lines:
        line = line.strip()
        # Skip common intro/header patterns
        if line and len(line) > 20 and not any(skip in line.lower() for skip in [
            'here are', 'following', 'claims:', 'verified:', 'decontextualized'
        ]):
            potential_claims.append(line)
    
    if potential_claims:
        return potential_claims
    
    # Last resort: return the whole content as a single item if it's substantial
    if content.strip() and len(content.strip()) > 10:
        return [content.strip()]
    
    # Empty response
    return []


def extract_process_name(summary: str) -> str:
    """Extract process name from summary, handling various formats."""
    # Try to find "Process: <name>" pattern
    process_match = re.search(r'Process:\s*(.+?)(?:\n|$)', summary, re.IGNORECASE)
    if process_match:
        return process_match.group(1).strip()

    # Fallback: try first line if it looks like a process name
    first_line = summary.split("\n")[0].strip()
    if first_line and len(first_line) < 200:  # Reasonable process name length
        return first_line

    # Last resort: return first 50 chars
    return summary[:50].strip()


def load_dataset(file_path: Path, id_column: str = None, genes_column: str = None):
    """
    Load dataset from CSV or TSV file with flexible column detection.
    
    Args:
        file_path: Path to the dataset file
        id_column: Name of ID column (auto-detected if None)
        genes_column: Name of genes column (auto-detected if None)
    
    Returns:
        DataFrame with standardized 'ID' and 'Genes' columns
    """
    # Detect file format
    if file_path.suffix.lower() == '.tsv':
        df = pd.read_csv(file_path, sep='\t', header=0)
    else:
        df = pd.read_csv(file_path, header=0)
    
    # Auto-detect columns if not specified
    if id_column is None:
        # Common ID column names
        id_candidates = ['ID', 'id', 'NEST ID', 'GeneSet_ID', 'geneSet_ID']
        id_column = next((col for col in id_candidates if col in df.columns), df.columns[0])
    
    if genes_column is None:
        # Common genes column names
        genes_candidates = ['Genes', 'genes', 'Gene', 'gene', 'Gene_Set', 'gene_set']
        genes_column = next((col for col in genes_candidates if col in df.columns), None)
        if genes_column is None:
            raise ValueError(f"Could not find genes column. Available columns: {list(df.columns)}")
    
    # Standardize column names
    df = df.rename(columns={id_column: 'ID', genes_column: 'Genes'})
    
    # Validate required columns exist
    if 'ID' not in df.columns or 'Genes' not in df.columns:
        raise ValueError(f"Required columns not found. Available: {list(df.columns)}")
    
    # Remove rows with missing data
    df = df.dropna(subset=['ID', 'Genes'])
    
    return df[['ID', 'Genes']]


def get_processed_ids(output_dir: Path) -> set:
    """Get set of already processed IDs from output files."""
    processed = set()
    final_file = output_dir / "Final_Response_GeneAgent.txt"

    if final_file.exists():
        with open(final_file, 'r') as f:
            for line in f:
                if line.startswith('['):
                    processed.add(line[1:].split(']')[0].strip())

    return processed


def GeneAgent(ID, genes, llm_model, dataset_name, output_dir: Path, resume: bool = False):
    """
    Run GeneAgent cascade workflow for a single gene set.

    Args:
        ID: Identifier for the gene set
        genes: Gene set string (comma-separated)
        llm_model: LLM model name
        dataset_name: Name of the dataset
        output_dir: Output directory path
        resume: If True, skip if already processed
    """
    # Initialize agentphd here (not at module level) to avoid hanging on import
    agentphd = AgentPhD(function_names=reposits)

    genes = genes.replace("/",",").replace(" ",",")

    pattern = re.compile(r'^[a-zA-Z0-9,.;?!*()_-]+$')

    # Check if already processed (for resume mode)
    if resume:
        processed_ids = get_processed_ids(output_dir)
        if str(ID) in processed_ids:
            print(f"Skipping {ID} (already processed)")
            return
    
    # Initialize LLM using unified utility
    try:
        llm_client = get_llm_client(llm_model)
        print(f"Initialized LLM: {llm_model} (source: {llm_client.source})")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise
    
    ## send genes to GPT-4 and generate the original template of process name and analysis
    try:
        # Track total usage across steps
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        # Specify output filenames
        baseline_file = output_dir / "Baseline_LLM_Responses.txt"
        topic_file = output_dir / "Claims_and_Verification_Topic.txt"
        analysis_file = output_dir / "Claims_and_Verification_Analytic_Narratives.txt"
        final_file = output_dir / "Final_Response_GeneAgent.txt"

        # Obtain the baseline summary
        print(f"=====Generating Baseline Summary for {ID}=====")
        prompt_baseline = baseline(genes)
        messages = [
            {"role":"system", "content":system},
            {"role":"user", "content":prompt_baseline}
        ]
        
        summary, usage_metrics = llm_client.chat(messages)
        
        # Track usage from BaseAgent's usage_metrics
        total_prompt_tokens, total_completion_tokens, total_cost, cost_info = _track_usage_and_cost(
            usage_metrics, llm_model, f"{dataset_name}_baseline_summary",
            total_prompt_tokens, total_completion_tokens, total_cost
        )
        print(f"$ Cost baseline: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")

        print("=====Saving Baseline Summary=====")
        with open(baseline_file,"a") as f_summary:
            f_summary.write(f"[{ID}]\n")
            f_summary.write(summary.strip()+"\n")
            f_summary.write("//\n")
        
        # send genes and process name to GPT-4 for topic verification.
        print("=====Generating Topic Claims/Process Names to Be Verified=====")
        process = extract_process_name(summary)
        prompt_topic = topic(genes, process) + topic_instruction
        message_topic = [
            {"role":"system", "content":system_verify},
            {"role":"user", "content":prompt_topic}
        ]
        
        claims_topic_content, usage_metrics = llm_client.chat(message_topic)
        
        # Track usage from BaseAgent's usage_metrics
        total_prompt_tokens, total_completion_tokens, total_cost, cost_info = _track_usage_and_cost(
            usage_metrics, llm_model, f"{dataset_name}_claims_topic",
            total_prompt_tokens, total_completion_tokens, total_cost
        )
        print(f"$ Cost topic claims: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")

        print("=====Saving Topic Claims/Process Names to Be Verified=====")
        # claims_topic = json.loads(claims_topic_content)
        claims_topic = extract_json_list(claims_topic_content)
        with open(topic_file,"a") as f_claim:
            f_claim.write(f"[{ID}]\n")
            f_claim.write(str(claims_topic)+"\n")
            f_claim.write("&&\n")
        
        print("=====Verifying Topic Claims/Process Names=====")
        verification_topic = ""
        for claim in claims_topic:
            if not re.match(pattern, claim):
                claim = re.sub(r'[^a-zA-Z0-9,.;?!*()_-]+$', "_", claim)
            claim_result = agentphd.inference(llm_model, claim)
            verification_topic += f"Original_claim:{claim}"
            verification_topic += f"Verified_claim:{claim_result}"
            with open(topic_file,"a") as f_claim:
                f_claim.write(str(claim)+"\n")
                f_claim.write(str(claim_result)+"\n")
                f_claim.write("&&\n")
            
        print("=====Updating Topic Claims/Process Names Based on Verification=====")
        modification_prompt = modification(verification_topic) + modification_instruction
        messages.append(
            {"role":"user", "content": modification_prompt}
            )
        updated_topic, usage_metrics = llm_client.chat(messages)
        messages.append({"role":"assistant", "content": updated_topic})
        
        # Track usage from BaseAgent's usage_metrics
        total_prompt_tokens, total_completion_tokens, total_cost, cost_info = _track_usage_and_cost(
            usage_metrics, llm_model, f"{dataset_name}_updated_topic",
            total_prompt_tokens, total_completion_tokens, total_cost
        )
        print(f"$ Cost updated topic: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})") 
        
        print("=====Generating Analysis Claims/Analytic Narratives to Be Verified=====")
        if not re.match(pattern, str(updated_topic)):
            updated_topic = re.sub(r'[^a-zA-Z0-9-_]+', "_", str(updated_topic))
        # send genes and updated summary to GPT-4 for analysis verification.
        prompt_analysis = analysis(updated_topic) + analysis_instruction
        analysis_message = [
            {"role":"system", "content":system_verify},
            {"role":"user", "content":prompt_analysis}
        ]
        claims_analysis_content, usage_metrics = llm_client.chat(analysis_message)
        
        # Track usage from BaseAgent's usage_metrics
        total_prompt_tokens, total_completion_tokens, total_cost, cost_info = _track_usage_and_cost(
            usage_metrics, llm_model, f"{dataset_name}_claims_analysis",
            total_prompt_tokens, total_completion_tokens, total_cost
        )
        print(f"$ Cost analysis claims: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")
        claims_analysis = extract_json_list(claims_analysis_content)

        print("=====Saving Analysis Claims/Analytic Narratives to Be Verified=====")
        with open(analysis_file,"a") as f_claim:
            f_claim.write(f"[{ID}]\n")
            f_claim.write(str(claims_analysis)+"\n")
            f_claim.write("&&\n")
        
        print("=====Verifying Analysis Claims/Analytic Narratives=====")
        verification_analysis = ""
        for claim in claims_analysis:
            if not re.match(pattern, claim):
                claim = re.sub(r'[^a-zA-Z0-9,.;?!*()_-]+$', "_", claim)
            claim_result = agentphd.inference(llm_model, str(claim))
            verification_analysis += f"Original_claim:{claim}"
            verification_analysis += f"Verified_claim:{claim_result}"
            with open(analysis_file, "a") as f_claim:
                f_claim.write(str(claim)+"\n")
                f_claim.write(str(claim_result)+"\n")
                f_claim.write("&&\n")
            
        ## send verificaton report to LLMs and modify the gene analysis
        print("=====Updating Analysis Claims/Analytic Narratives Based on Verification=====")
        summarization_prompt = summarization(verification_analysis) + summarization_instruction
        messages.append(
            {"role":"user", "content":summarization_prompt }
        )
        update, usage_metrics = llm_client.chat(messages)
        
        # Track usage from BaseAgent's usage_metrics
        total_prompt_tokens, total_completion_tokens, total_cost, cost_info = _track_usage_and_cost(
            usage_metrics, llm_model, f"{dataset_name}_final_update",
            total_prompt_tokens, total_completion_tokens, total_cost
        )
        print(f"$ Cost final update: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")

        with open(final_file,"a") as f_final:
            f_final.write(f"[{ID}]\n")
            f_final.write(update.strip()+"\n")
            f_final.write("//\n")
                
        # Write a totals line for this ID
        totals_file = output_dir / "Usage_Totals.txt"
        with open(totals_file, "a") as f_totals:
            f_totals.write(f"{dataset_name}\t{ID}\t{total_prompt_tokens}\t{total_completion_tokens}\t{total_cost:.4f}\n")

        print(f"=== Totals for {dataset_name} {ID}: in={total_prompt_tokens}, out={total_completion_tokens}, cost=${total_cost:.4f}")

        with open(analysis_file,"a") as f_claim:
            f_claim.write("////\n")

    except Exception as E:
        error_file = output_dir / "Error_Report.txt"
        with open(error_file,"a") as f_final:
            f_final.write(f"{ID}\t")
            f_final.write(f"====There is an error {E} here.====\n")
            f_final.write("//\n")
                
        print(f"====There is an error {E} here.====")
        raise  # Re-raise to allow caller to handle


def main():
    parser = argparse.ArgumentParser(
        description="GeneAgent: Self-verification Language Agent for Gene Set Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main_cascade.py --input Datasets/MsigDB/MsigDB_toy.csv --llm gpt-4o

  # With custom output directory
  python main_cascade.py --input Datasets/GO/GO_toy.csv --llm gpt-4o --output ./results

  # Resume from previous run
  python main_cascade.py --input Datasets/MsigDB/MsigDB_toy.csv --llm gpt-4o --resume

  # TSV file with custom columns
  python main_cascade.py --input Datasets/NeST/NeST_toy.tsv --id-column "NEST ID" --genes-column "Genes" --llm gpt-4o

  # Using Ollama with memory cleanup (prevents OOM crashes)
  python main_cascade.py --input Datasets/MsigDB/MsigDB_toy.csv --llm gpt-oss:20b --cleanup-memory

        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input dataset file (CSV or TSV)'
    )
    
    parser.add_argument(
        '--llm', '-l',
        type=str,
        default='gpt-4o',
        help='LLM model name (default: gpt-4o). Options: gpt-5,gpt-4o, azure-gpt-4o, gpt-oss:20b'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: Outputs/{llm}/{dataset_name})'
    )
    
    parser.add_argument(
        '--id-column',
        type=str,
        default=None,
        help='Name of ID column (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--genes-column',
        type=str,
        default=None,
        help='Name of genes column (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume processing, skipping already processed gene sets'
    )
    
    parser.add_argument(
        '--clear-output',
        action='store_true',
        help='Clear existing output files before starting'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of gene sets to process (for testing)'
    )

    parser.add_argument(
        '--cleanup-memory',
        action='store_true',
        help='Unload Ollama model from memory after each gene set to prevent OOM errors'
    )


    args = parser.parse_args()
    
    # Resolve paths
    input_file = Path(args.input).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    dataset_name = input_file.stem
    
    # Set up output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        base_dir = Path(__file__).absolute().parent
        output_dir = base_dir / "Outputs" / args.llm / dataset_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear output files if requested
    if args.clear_output and not args.resume:
        output_files = [
            output_dir / "Baseline_LLM_Responses.txt",
            output_dir / "Claims_and_Verification_Topic.txt",
            output_dir / "Claims_and_Verification_Analytic_Narratives.txt",
            output_dir / "Final_Response_GeneAgent.txt",
            output_dir / "Usage_Totals.txt",
            output_dir / "Error_Report.txt",
        ]
        for f in output_files:
            f.unlink(missing_ok=True)
        print("Cleared existing output files")
    
    # Load dataset
    print(f"Loading dataset from: {input_file}")
    try:
        df = load_dataset(input_file, args.id_column, args.genes_column)
        print(f"Loaded {len(df)} gene sets")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Limit if specified
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} gene sets for processing")
    
    # Process each gene set
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        ID = row['ID']
        genes = row['Genes']

        print(f"\n{'='*60}")
        print(f"Processing {idx}/{total}: {ID}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        try:
            # In main_cascade.py after processing each gene set
            llm_client = get_llm_client(args.llm)
            GeneAgent(ID, genes, args.llm, dataset_name, output_dir)
            # Cleanup memory after processing if requested
            llm_client.cleanup_memory() 
            # print time after processing each gene set
            print(f"\n{'='*60}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Error processing {ID}: {e}")
            if not args.resume:
                # In non-resume mode, continue with next item
                continue
            else:
                # In resume mode, stop on error to allow fixing
                raise

    print("\n===Finished!===")

    # Force cleanup and exit
    sys.stdout.flush()
    sys.stderr.flush()


def _force_cleanup():
    """Force cleanup of resources on exit."""
    try:
        from llm_utils import cleanup_all_clients
        cleanup_all_clients()
    except:
        pass


if __name__ == "__main__":
    # Register cleanup handler
    atexit.register(_force_cleanup)

    try:
        main()
    finally:
        # Ensure cleanup runs even if main() fails
        _force_cleanup()
        # Force flush output streams
        sys.stdout.flush()
        sys.stderr.flush()
        # Exit explicitly with success code
        sys.exit(0)
