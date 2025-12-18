import time
import json
import re
import uuid

from llm_utils import get_llm_client
from costs import record_chat_completion_cost


def repair_json_string(json_str: str) -> str:
	"""
	Repair common JSON formatting errors from Ollama models.

	Common issues:
	- Extra closing braces: {"key": "val"}}
	- Trailing commas: {"key": "val",}
	- Missing quotes: {key: "val"}
	- Extra spaces/newlines
	"""
	# Remove extra whitespace
	json_str = json_str.strip()

	# Remove trailing commas before closing braces
	json_str = re.sub(r',\s*}', '}', json_str)
	json_str = re.sub(r',\s*]', ']', json_str)

	# Remove extra closing braces at the end
	while json_str.endswith('}}') and json_str.count('{') < json_str.count('}'):
		json_str = json_str[:-1]

	# Try to balance braces
	open_braces = json_str.count('{')
	close_braces = json_str.count('}')
	if open_braces > close_braces:
		json_str += '}' * (open_braces - close_braces)
	elif close_braces > open_braces:
		# Remove extra closing braces from the end
		extra = close_braces - open_braces
		for _ in range(extra):
			if json_str.endswith('}'):
				json_str = json_str[:-1]

	return json_str


def extract_and_repair_tool_call(error_message: str, available_functions: dict = None) -> dict:
	"""
	Extract tool call arguments from Ollama error message and attempt to repair them.

	Error format: "error parsing tool call: raw='{"gene..."}', err=..."
	"""
	# Extract the raw JSON from the error message
	match = re.search(r"raw='([^']+)'", error_message)
	if not match:
		return None

	raw_json = match.group(1)

	# Attempt to repair common JSON issues
	repaired_json = repair_json_string(raw_json)

	try:
		# Try to parse the repaired JSON
		parsed_args = json.loads(repaired_json)

		# Try to infer function name from arguments
		function_name = "unknown"
		if available_functions:
			# Match based on parameter names
			arg_keys = set(parsed_args.keys())
			for func_name, func_schema in available_functions.items():
				params = func_schema.get("parameters", {}).get("properties", {})
				param_keys = set(params.keys())
				# If all required args match, this is likely the right function
				if arg_keys.issubset(param_keys) or param_keys.issubset(arg_keys):
					function_name = func_name
					break

		# Construct a tool call in LangChain format
		tool_call = {
			"name": function_name,
			"args": parsed_args,
			"id": f"call_{uuid.uuid4().hex[:24]}"
		}
		return tool_call
	except json.JSONDecodeError as e:
		print(f"✗ Failed to repair JSON: {e}")
		return None

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
		llm_client = get_llm_client(llm_model, temperature)

		# Initialize system message and user message
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
				# Try to repair Ollama JSON parsing errors
				if "error parsing tool call" in error_msg or "invalid character" in error_msg:
					print(f"⚠ Ollama JSON error detected, attempting repair...")

					# Build a dict of available functions with their schemas
					available_funcs = {doc["name"]: doc for doc in self.function_docs}

					# Try to extract and repair the tool call from error message
					repaired_tool_call = extract_and_repair_tool_call(error_msg, available_funcs)

					if repaired_tool_call:
						print(f"✓ Repaired tool call: {repaired_tool_call['name']}")

						# Create a mock response object with the repaired tool call
						class MockResponse:
							def __init__(self, tool_calls):
								self.tool_calls = tool_calls
								self.content = ""

						response = MockResponse([repaired_tool_call])
						usage_metrics = None  # No usage metrics from failed call
					else:
						print(f"✗ Could not repair JSON, skipping iteration")
						continue
				else:
					# Re-raise unexpected errors
					raise

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

				# Execute ALL tool calls (GPT-4o often makes multiple calls in parallel)
				for tool_call in response.tool_calls:
					try:
						function_name = tool_call["name"]
						function_params = tool_call["args"]
						function_to_call = self.name2function[function_name]
						function_response = function_to_call(**function_params)
						function_response = f"Function has been called with params {function_params}, and returns {function_response}."

						# Add tool response
						message_verification.append({
							"role": "tool",
							"tool_call_id": tool_call["id"],
							"content": function_response
						})

					except Exception as E:
						# Add error response
						message_verification.append({
							"role": "tool",
							"tool_call_id": tool_call["id"],
							"content": f"Function has been called with params {function_params}, but returned error: {E}. Please try again with the correct parameter."
						})

			else:
				try:
					if response.content and "Report: " in response.content:
						report = response.content.split("Report: ")[-1]
						if re.match(pattern, report):
							return report
						else:
							return re.sub(r'[^a-zA-Z0-9_-]+$', "_", report)

					else:
						# Ask for final report
						message_verification.append({
							"role": "user",
							"content": f"please start a message with \"Report:\" and return your findings if you have obtained the verification information.",
						})

				except Exception as E:
					message_verification.append({
						"role": "assistant",
						"content": f"Claim has been verified, but returned error: {E}. Please try it again.",
					})

		return "Failed."
