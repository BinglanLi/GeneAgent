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

import time
import json
import re

import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

import tiktoken
MAX_TOKENS = 127900
encoding = tiktoken.encoding_for_model("gpt-4")

from apis.get_complex_for_gene_set import get_complex_for_gene_set, get_complex_for_gene_set_doc 
from apis.get_disease_for_single_gene import get_disease_for_single_gene, get_disease_for_single_gene_doc
from apis.get_domain_for_single_gene import get_domain_for_single_gene, get_domain_for_single_gene_doc
from apis.get_enrichment_for_gene_set import get_enrichment_for_gene_set, get_enrichment_for_gene_set_doc
from apis.get_pathway_for_gene_set import get_pathway_for_gene_set, get_pathway_for_gene_set_doc  
from apis.get_interactions_for_gene_set import get_interactions_for_gene_set, get_interactions_for_gene_set_doc 
from apis.get_gene_summary_for_single_gene import get_gene_summary_for_single_gene, get_gene_summary_for_single_gene_doc
 
from apis.get_pubmed_articles import get_pubmed_articles, get_pubmed_articles_doc

func2info = {
    "get_complex_for_gene_set": [get_complex_for_gene_set, get_complex_for_gene_set_doc],
	"get_disease_for_single_gene": [get_disease_for_single_gene, get_disease_for_single_gene_doc],
	"get_domain_for_single_gene": [get_domain_for_single_gene, get_domain_for_single_gene_doc],
	"get_enrichment_for_gene_set": [get_enrichment_for_gene_set, get_enrichment_for_gene_set_doc],
	"get_pathway_for_gene_set": [get_pathway_for_gene_set, get_pathway_for_gene_set_doc],
	"get_interactions_for_gene_set": [get_interactions_for_gene_set, get_interactions_for_gene_set_doc],
	"get_gene_summary_for_single_gene": [get_gene_summary_for_single_gene, get_gene_summary_for_single_gene_doc],
	"get_pubmed_articles": [get_pubmed_articles, get_pubmed_articles_doc]
}

pattern = re.compile(r'^[a-zA-Z0-9_-]+$')

class AgentPhD:
	def __init__(self, function_names):
		self.name2function = {function_name: func2info[function_name][0] for function_name in function_names}
		self.function_docs = [func2info[function_name][1] for function_name in function_names]

	def inference(self, claim):
    
		system = f"""
  		You are a helpful fact-checker. 
   		Your task is to verify the claim using the provided tools. 
     	If there are evidences in your contents, please start a message with "Report:" and return your findings along with evidences.
    	"""
		content = f"""
  		Here is the claim needed to be verified:\n{claim} 
		Try to use multiple tools to verify a claim and the verification process should be factual and objective.
    	Put your decision at the beginning of the evidences.
    	Don't use any format symbols such as '*', '-' or other tokens.
    	"""
		token_verification = encoding.encode(content + system)
		print(f"=====The prompt tokens input to the verification step is {len(token_verification)}=====")
		message_verification = [
			{"role": "system", "content": system},
			{"role": "user", "content": content} 
		]

		loop = 0
		while loop < 20:
			loop += 1
			# logger.info(f"Input@{loop}\n" +  json.dumps(messages, indent=4))
			time.sleep(1)
			completion = client.chat.completions.create(
				model="gpt-4o",
				messages=message_verification,
				functions=self.function_docs,
				temperature=0,
			)

			message = completion.choices[0].message
			cost_info = record_chat_completion_cost(completion, "gpt-4o", tag="verification_loop")
			print(f"$ Cost verification: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")
			# token_message_output = encoding.encode(str(message))
			# print(f"=====The message tokens output from the verification step is {len(token_message_output)}=====")
			# logger.info(f"Output@{loop}\n" +  json.dumps(message, indent=4))

			if getattr(message, "function_call", None):
				try:
					function_name = message.function_call.name
					function_params = json.loads(message.function_call.arguments)
					function_to_call = self.name2function[function_name]
					function_response = function_to_call(**function_params)
					function_response = f"Function has been called with params {function_params}, and returns {function_response}."

					message_verification.append(
						{
							"role": "function",
							"name": function_name,
							"content": function_response
						},
					)
					# token_message_verification = encoding.encode(str(message_verification))
					# print(f"=====The message tokens input to verification step is {len(token_message_verification)}=====")

				except Exception as E:
					message_verification.append(
						{
							"role": "function",
							"name": function_name,
							"content": f"Function has been called with params {function_params}, but returned error: {E}. Please try again with the correct parameter.",
						}
					)
					# token_message_verification = encoding.encode(str(message_verification))
					# print(f"=====The message tokens input to verification step is {len(token_message_verification)}=====")
			
			else:
				try:
					if message and getattr(message, "content", None) and "Report: " in message.content:
						report = message.content.split("Report: ")[-1]
						token_report = encoding.encode(report)
						print(f"=====The output tokens of verification report in the verification step is {len(token_report)}=====")
						if re.match(pattern, report):
							return report
						else: 
							return re.sub(r'[^a-zA-Z0-9_-]+$', "_", report)
					
					else:
						message_verification.append(
							{
								"role": "user",
								"content": f"please start a message with \"Report:\" and return your findings if you have obtained the verification information.",
							}
						)
						# token_message_verification = encoding.encode(str(message_verification))
						# print(f"=====The message tokens input to verification step is {len(token_message_verification)}=====")
      
				except Exception as E:
					message_verification.append(
						{
							"role": "assistant",
							"content": f"Claim has been verified, but returned error: {E}. Please try it again.",
						}
					)
					# token_message_verification = encoding.encode(str(message_verification))
					# print(f"=====The message tokens input to verification step is {len(token_message_verification)}=====")
					# print(E)

		return "Failed."	
