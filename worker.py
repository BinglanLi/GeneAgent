import time
import json
import re

from llm_utils import get_llm_client
from costs import record_chat_completion_cost

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
	"""
	Agent for verifying gene-related claims using function calling.
	Refactored to use SimpleLLMClient with BaseAgent infrastructure.
	"""
	
	def __init__(self, function_names):
		self.name2function = {function_name: func2info[function_name][0] for function_name in function_names}
		self.function_docs = [func2info[function_name][1] for function_name in function_names]

	def inference(self, llm_model, claim):
		"""
		Verify a claim using tool calling via BaseAgent's LangChain infrastructure.

		This method uses LangChain's native tool binding for consistent behavior
		across all providers (OpenAI, Anthropic, Ollama, etc.).
		"""
		# Determine temperature based on model
		temperature = 1.0 if llm_model.startswith("gpt-5") else 0
		# Get LLM client
		llm_client = get_llm_client(llm_model)
    
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
		message_verification = [
			{"role": "system", "content": system},
			{"role": "user", "content": content}
		]

		# Determine temperature: gpt-5 requires temperature=1, others use 0
		temperature = 1.0 if llm_model.startswith("gpt-5") else 0
		use_tool_calling = llm_model.startswith("gpt-5")  # gpt-5 uses tool calling format
		
		loop = 0
		while loop < 20:
			loop += 1
			time.sleep(1)

			# Use LangChain-native tool calling via BaseAgent
			try:
				response, usage_metrics = llm_client.chat_with_tools(
					messages=message_verification,
					tools=self.function_docs,
				)
			except Exception as E:
				error_msg = str(E)

			# Convert usage_metrics to dict for cost tracking
			usage_dict = None
			if usage_metrics:
				usage_dict = {
					"input_tokens": usage_metrics.input_tokens or 0,
					"output_tokens": usage_metrics.output_tokens or 0,
					"total_tokens": usage_metrics.total_tokens or 0,
				}

			# Record costs using BaseAgent's usage metrics
			cost_info = record_chat_completion_cost(model=llm_model, tag="verification_loop", usage_dict=usage_dict)
			print(f"$ Cost verification: ${cost_info['total_cost']:.4f} (in={cost_info['prompt_tokens']}, out={cost_info['completion_tokens']})")

			# Check if LLM made tool calls
			if hasattr(response, "tool_calls") and response.tool_calls:
				# Add the assistant's response with tool calls to history
				# Keep tool_calls in LangChain format for consistency
				assistant_msg = {
					"role": "assistant",
					"content": response.content or "",
					"tool_calls": response.tool_calls
				}
				message_verification.append(assistant_msg)

			if function_call:
				try:
					function_name = function_call.name
					function_params = json.loads(function_call.arguments)
					function_to_call = self.name2function[function_name]
					function_response = function_to_call(**function_params)
					function_response = f"Function has been called with params {function_params}, and returns {function_response}."

					if use_tool_calling:
						# gpt-5 uses tool role with tool_call_id
						message_verification.append(
							{
								"role": "tool",
								"tool_call_id": tool_call_id,
								"content": function_response
							},
						)
					else:
						# Legacy function role
						message_verification.append(
							{
								"role": "function",
								"name": function_name,
								"content": function_response
							},
						)

				except Exception as E:
					if use_tool_calling:
						message_verification.append(
							{
								"role": "tool",
								"tool_call_id": tool_call_id,
								"content": f"Function has been called with params {function_params}, but returned error: {E}. Please try again with the correct parameter.",
							}
						)
					else:
						message_verification.append(
							{
								"role": "function",
								"name": function_name,
								"content": f"Function has been called with params {function_params}, but returned error: {E}. Please try again with the correct parameter.",
							}
						)
			
			else:
				try:
					if message and getattr(message, "content", None) and "Report: " in message.content:
						report = message.content.split("Report: ")[-1]
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
      
				except Exception as E:
					message_verification.append(
						{
							"role": "assistant",
							"content": f"Claim has been verified, but returned error: {E}. Please try it again.",
						}
					)

		return "Failed."
