import json
import time
import pandas as pd

from openai import OpenAI
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None
import os
from dotenv import load_dotenv
load_dotenv()

from costs import record_chat_completion_cost

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
from topic import topic_verification

if __name__ == "__main__":
    
    ## baseline 
    system = "You are an efficient and insightful assistant to a molecular biologist."
    task = lambda genes: f"Your task is to propose a biological process term for gene sets. Here is the gene set: {genes}"
    chain = f"""
    Let do the task step-by-step:
    Step1, write a cirtical analysis for gene functions. For each important point, discribe your reasoning and supporting information.
    Step2, analyze the functional associations among different genes from the critical analysis.
    Step3, summarize a brief name for the most significant biological process of gene set from the functional associations. 
    """
    instruction = """
    Put the name at the top of analysis as "Process: <name>" and follow the analysis.
    Be concise, do not use unnecessary words.
    Be specific, avoid overly general statements such as "the proteins are involved in various cellular processes".
    Be factual, do not editorialize.
    """
    
    data = pd.read_csv("Datasets/MsigDB/MsigDB.csv", header=0, index_col=None)
    for genes in data["Genes"]:
        genes = genes.replace(" ",",")
        ## send genes to GPT-4 and generate the original template of process name and analysis
        prompt_baseline = task(genes) + chain + instruction
        messages = [
            {"role":"system", "content":system},
            {"role":"user", "content":prompt_baseline}
        ]
        summary = client.chat.completions.create(
			model="gpt-4o",
			messages=messages,
			temperature=0.0,
		)
        messages.append(summary.choices[0].message)
        cost_info = record_chat_completion_cost(summary, "gpt-4o", tag="cot_summary")
        print(f"$ Cost CoT: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")
        summary = summary.choices[0].message.content
        with open("Outputs/Chain-of-Thought/MsigDB_Response_CoT.txt","a") as f_update:
            f_update.write(summary+"\n")
            f_update.write("//\n")
        print("=====Summary=====")
        print(summary)
        
        
