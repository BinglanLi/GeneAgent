import json
import requests
from ._session import make_session

def get_gene_summary_for_single_gene(gene_name, specie):
	base_url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
	base_url_summary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
	term = gene_name + " AND " + specie
	search_params = {
		"db": "gene",
		"term": term,
		"retmode": "json",
		"sort": "relevance"
	}
	try:
		session = make_session()
		search_response = session.get(base_url_search, params=search_params, timeout=30)
		gene_id = search_response.json().get('esearchresult', {}).get('idlist', [])

		if gene_id:
			summary_params = {
				"db": "gene",
				"id": gene_id[0],
				"retmode": "json",
				"sort": "relevance"
			}
			summary_response = session.get(base_url_summary, params=summary_params, timeout=30)
			gene_summaries = summary_response.json().get('result', {})[gene_id[0]]
			gene_summaries.pop('locationhist')
			return gene_summaries

		return None
	except requests.exceptions.RequestException as e:
		return f"Error: {e}"
		

get_gene_summary_for_single_gene_doc = {
	"name": "get_gene_summary_for_single_gene",
	"description": "Given a single gene name, return summary information on function and so on.",
	"parameters": {
		"type": "object",
		"properties": {
			"gene_name": {
				"type": "string",
				"description": "A single gene name to search.",
			},
   			"specie": {
				"type": "string",
				"description": "A specie name to search. Only have the human specie (Homo) and mouse specie (Mus) now.",
				"enum": ["Homo","Mus"]
			},
		},
		"required": ["gene_name","specie"],
	},
}
