import requests
import json
from ._session import make_session

def get_disease_for_single_gene(gene_name):
    url = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/agentapi/disease/?"
    params = {
        "name": gene_name,
        "retmode": "json",
        "limit": 100
        }
    try:
        session = make_session()
        response = session.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return json.dumps(response.json().get("results", {}))
        return f"Error: Unable to fetch data (Status Code: {response.status_code})"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Example usage
# gene_name = "BRCA1"  # Replace with the gene name you are interested in
# gene_info = get_gene_complex(gene_name)
# print(gene_info)

get_disease_for_single_gene_doc = {
	"name": "get_disease_for_single_gene",
	"description": "Given a gene name, return information on the related diseases containing the disease id and the corresponding disease name.",
	"parameters": {
		"type": "object",
		"properties": {
			"gene_name": {
				"type": "string",
				"description": "A single gene name to search."
                }
            },
		"required": ["gene_name"],
	},
}


