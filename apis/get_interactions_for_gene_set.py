import requests
import json
from ._session import make_session

def get_interactions_for_gene_set(gene_set):
    url = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/agentapi/ppi/?"
    params = {
        "name": gene_set,
        "retmode": "json",
        "limit": 50
        }
    try:
        session = make_session()
        response = session.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return json.dumps(response.json().get("results", {}))
        return f"Error: Unable to fetch data (Status Code: {response.status_code})"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

get_interactions_for_gene_set_doc = {
	"name": "get_interactions_for_gene_set",
	"description": "Given a gene set, return information on its interacting genes.",
	"parameters": {
		"type": "object",
		"properties": {
			"gene_set": {
				"type": "string",
				"description": "A gene set delimitted with comma to search, for example, \"x,y,z\"."
                }
            },
		"required": ["gene_set"],
	},
}


