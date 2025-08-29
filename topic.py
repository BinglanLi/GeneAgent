import json
import time
import re
import pandas as pd

from openai import OpenAI
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None
import os
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


## topic verification
system_verify = "You are a helpful and objective fact-checker to verify the process name of gene set."
topic = lambda genes, process: f"""
Here is the vanilla process name for the human gene set {genes}:\n{process}
However, the process name might be false. Please generate decontextualized claims for the process name that need to be verified.
Please return JSON list only containing the generated strings of claims:
"""
topic_instruction = """
Generate claims of affirmative sentences about the prominent biological process for the entire gene set.
Don't generate negative sentences in claims for the entire gene set.
Don't generate claims for the single gene or incomplete gene set.
Don't generate hypotheis claims over the previous analysis like diseases, mutations, disruptions, etc.
Please replace the statement like 'these genes', 'this system' with the entire gene set.
"""

def topic_verification(genes, process_name, agentphd):  
    pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
    ## send genes and summary to GPT-4 and generate claims for verifying topic name
    prompt_topic = topic(genes, process_name) + topic_instruction
    message = [
        {"role":"system", "content":system_verify},
        {"role":"user", "content":prompt_topic}
    ]
    claims = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        temperature=0.0,
        )
    cost_info = record_chat_completion_cost(claims, "gpt-4o", tag="topic_claims")
    print(f"$ Cost topic claims: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")
    claims = json.loads(claims.choices[0].message.content)
    print("=====Topic Claim=====")
    print(claims)
    
    verification = ""
    for claim in claims:
        if not re.match(pattern, claim):
            claim = re.sub(r'[^a-zA-Z0-9,.;?!*()_-]+$', "_", claim)
        claim_result = agentphd.inference(claim)
        verification += f"Original_claim:{claim}"
        verification += f"Verified_claim:{claim_result}"
        with open("Outputs/Verification Reports/Synchronous/Claims_and_Verification_for_MsigDB.txt","a") as f_claim:
            f_claim.write(str(claim)+"\n")
            f_claim.write(str(claim_result)+"\n")
            f_claim.write("&&\n")
        print(claim)
        print(claim_result)
        
    ## send verificaton report to GPT-4 and modify the original process name
    message.append(
        {"role":"assistant", "content":f"There should be only one most significant function name. If the process name is direclty supported in all verifications, the significant function is the name that most similar to the original process name but reflects more specific biological regulation mechanism. Otherwise, it is the first (top-1) function name in verifications."}
    )
    message.append(
        {"role":"user", "content":f"I have finished the verification for the process name, here is the verification report:{verification}\nPlease replace the process name with the most significant function of gene set.\nPlease start a message with \"Topic:\" and only return the brief revised name."}
    )
    updated = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        temperature=0.0,
        )
    cost_info = record_chat_completion_cost(updated, "gpt-4o", tag="topic_update")
    print(f"$ Cost topic update: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")

    # messages.append(updated_topic.choices[0]["message"])
    updated = updated.choices[0].message.content

    print("=====Updated Topic=====")
    print(updated)
    
    return updated
