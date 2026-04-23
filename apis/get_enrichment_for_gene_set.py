import json
import requests
from ._session import make_session

def get_enrichment_for_gene_set(gene_set):

    gene_set = gene_set.replace(" ","")
    gene_list = gene_set.split(",")

    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "organism": "hsapiens",
        "query": gene_list,
        "sources": [],
        "all_results": False,
        "user_threshold": 0.05
    }

    try:
        session = make_session()
        response = session.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if response.status_code == 200:
            return json.dumps(response.json()["result"][:5])
        return f"Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

get_enrichment_for_gene_set_doc = {
    "name": "get_enrichment_for_gene_set",
    "description": "Given a gene set only separated by \",\", return its top-5 enrichment function names containing biological regulation, signaling, and metabolism.",
        "parameters": {
        "type": "object",
        "properties": {
            "gene_set": {
                "type": "string",
                "description": "A gene set separated by only \",\" (must no whitespace) to search. For example: \"x,y,z\".",
            },
        },
        "required": ["gene_set"],
    },
}
