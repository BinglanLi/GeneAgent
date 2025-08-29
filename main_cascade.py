import json
import re
import time
from turtle import up
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
from openai import OpenAI
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None
from costs import record_chat_completion_cost

load_dotenv()

def _create_openai_client():
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_API_BASE")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_API_VERSION")
    if azure_endpoint and azure_api_key and azure_api_version and AzureOpenAI is not None:
        return AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
        )
    return OpenAI()

client = _create_openai_client()

from worker import AgentPhD

import tiktoken
MAX_TOKENS = 127900
encoding = tiktoken.encoding_for_model("gpt-4o")

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


agentphd = AgentPhD(function_names=reposits)

def GeneAgent(ID, genes):    
    genes = genes.replace("/",",").replace(" ",",")
    
    pattern = re.compile(r'^[a-zA-Z0-9,.;?!*()_-]+$')
    ## send genes to GPT-4 and generate the original template of process name and analysis
    try:
        # Ensure output directories exist
        os.makedirs("Outputs/GPT-4", exist_ok=True)
        os.makedirs("Outputs/GeneAgent/Cascade", exist_ok=True)
        os.makedirs("Outputs/Verification Reports/Cascade", exist_ok=True)
        prompt_baseline = baseline(genes)
        first_step = prompt_baseline + system
        token_baseline = encoding.encode(first_step)
        print(f"=====The prompt tokens input to the generation step is {len(token_baseline)}=====\n")
        messages = [
            {"role":"system", "content":system},
            {"role":"user", "content":prompt_baseline}
        ]
        summary_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        messages.append(summary_resp.choices[0].message)
        summary = summary_resp.choices[0].message.content
        cost_info = record_chat_completion_cost(summary_resp, "gpt-4o", tag="baseline_summary")
        print(f"$ Cost baseline: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")

        with open("Outputs/GPT-4/Baseline_LLM_Responses.txt","a") as f_summary:
            f_summary.write(summary+"\n")
            f_summary.write("//\n")
        print("=====Summary=====")
        print(summary)
        
        # send genes and process name to GPT-4 for topic verification.
        process = summary.split("\n")[0].split("Process: ")[1]
        prompt_topic = topic(genes, process) + topic_instruction
        message_topic = [
            {"role":"system", "content":system_verify},
            {"role":"user", "content":prompt_topic}
        ]
        claims_topic_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=message_topic,
            temperature=0,
        )
        cost_info = record_chat_completion_cost(claims_topic_resp, "gpt-4o", tag="claims_topic")
        print(f"$ Cost topic claims: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")
        claims_topic = json.loads(claims_topic_resp.choices[0].message.content)
        with open("Outputs/Verification Reports/Cascade/Claims_and_Verification_Topic.txt","a") as f_claim:
            f_claim.write(str(claims_topic)+"\n")
            f_claim.write("&&\n")
        print("=====Topic Claim=====")
        print(claims_topic)
        
        verification_topic = ""
        for claim in claims_topic:
            if not re.match(pattern, claim):
                claim = re.sub(r'[^a-zA-Z0-9,.;?!*()_-]+$', "_", claim)
            claim_result = agentphd.inference(claim)
            verification_topic += f"Original_claim:{claim}"
            verification_topic += f"Verified_claim:{claim_result}"
            with open("Outputs/Verification Reports/Cascade/Claims_and_Verification_Topic.txt","a") as f_claim:
                f_claim.write(str(claim)+"\n")
                f_claim.write(str(claim_result)+"\n")
                f_claim.write("&&\n")
            print(claim)
            print(claim_result)
            
        modification_prompt = modification(verification_topic) + modification_instruction
        messages.append(
            {"role":"user", "content": modification_prompt}
            )
        updated_topic_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        messages.append(updated_topic_resp.choices[0].message)
        cost_info = record_chat_completion_cost(updated_topic_resp, "gpt-4o", tag="updated_topic")
        print(f"$ Cost updated topic: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")
        updated_topic = updated_topic_resp.choices[0].message.content 
        print("=====Updated Topic=====")
        print(updated_topic)
        
        if not re.match(pattern, str(updated_topic)):
            updated_topic = re.sub(r'[^a-zA-Z0-9-_]+', "_", str(updated_topic))
        # send genes and updated summary to GPT-4 for analysis verification.
        prompt_analysis = analysis(updated_topic) + analysis_instruction

        analysis_message = [
            {"role":"system", "content":system_verify},
            {"role":"user", "content":prompt_analysis}
        ]
        claims_analysis_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=analysis_message,
            temperature=0,
        )
        cost_info = record_chat_completion_cost(claims_analysis_resp, "gpt-4o", tag="claims_analysis")
        print(f"$ Cost analysis claims: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")
        claims_analysis = json.loads(claims_analysis_resp.choices[0].message.content)
        with open("Outputs/Verification Reports/Cascade/Claims_and_Verification_Analytic_Narratives.txt","a") as f_claim:
            f_claim.write(str(claims_analysis)+"\n")
            f_claim.write("&&\n")
        print("=====Analysis Claim=====")
        print(claims_analysis)
        
        verification_analysis = ""
        for claim in claims_analysis:
            if not re.match(pattern, claim):
                claim = re.sub(r'[^a-zA-Z0-9,.;?!*()_-]+$', "_", claim)
            claim_result = agentphd.inference(str(claim))
            verification_analysis += f"Original_claim:{claim}"
            verification_analysis += f"Verified_claim:{claim_result}"
            with open("Outputs/Verification Reports/Cascade/Claims_and_Verification_Analytic_Narratives.txt","a") as f_claim:
                f_claim.write(str(claim)+"\n")
                f_claim.write(str(claim_result)+"\n")
                f_claim.write("&&\n")
            print(claim)
            print(claim_result)
            
        ## send verificaton report to GPT-4 and modify the gene analysis
        summarization_prompt = summarization(verification_analysis) + summarization_instruction
        messages.append(
            {"role":"assistant", "content":summarization_prompt }
        )
        updated_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        cost_info = record_chat_completion_cost(updated_resp, "gpt-4o", tag="final_update")
        print(f"$ Cost final update: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")
        update = updated_resp.choices[0].message.content

        with open("Outputs/GeneAgent/Cascade/Final_Response_GeneAgent.txt","a") as f_final:
            f_final.write(update+"\n")
            f_final.write("//\n")
        print("====Final Update====")
        print(update)
                
        with open("Outputs/Verification Reports/Cascade/Claims_and_Verification_for_MsigDB.txt","a") as f_claim:
            f_claim.write("////\n")

    except Exception as E:
        with open("Outputs/GeneAgent/Cascade/Error_Report.txt","a") as f_final:
            f_final.write(ID + "\t")
            f_final.write(f"====There are an error {E} here.====\n")
            f_final.write("//\n")
                
        print(f"====There are an error {E} here.====")       

            
if __name__ == "__main__":

    data = pd.read_csv("Datasets/AlzKB/gene_sets.csv", header=0, index_col=None)
    for ID, genes in zip(data["ID"], data["Genes"]):
        GeneAgent(ID, genes)
        
    print("===Finished!===")
    
