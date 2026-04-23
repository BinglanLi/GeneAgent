import json
import requests
from ._session import make_session

def get_pathway_for_gene_set(gene_set):
    gene_set = gene_set.replace(" ","")
    gene_list = gene_set.split(",")

    ENRICHR_URL_ADD = 'http://maayanlab.cloud/Enrichr/addList'
    payload = {
        'list': (None, '\n'.join(gene_list)),
        'description': (None, 'My gene set')
    }
    try:
        session = make_session()
        response_add = session.post(ENRICHR_URL_ADD, files=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        raise Exception(f'Error adding list to Enrichr: {e}')

    if not response_add.ok:
        raise Exception('Error adding list to Enrichr:', response_add.text)

    data = json.loads(response_add.text)
    list_id = data['userListId']
    dic = {}
    for backgroundType in ["KEGG_2021_Human", "Reactome_2022", "BioPlanet_2019", "MSigDB_Hallmark_2020"]:
        try:
            ENRICHR_URL_RESULTS = f'http://maayanlab.cloud/Enrichr/enrich?userListId={list_id}&backgroundType={backgroundType}'
            try:
                response_results = session.get(ENRICHR_URL_RESULTS, timeout=30)
            except requests.exceptions.RequestException as e:
                raise Exception(f'Error fetching pathway results: {e}')

            if not response_results.ok:
                raise Exception('Error fetching pathway results:', response_results.text)

            pathway_data = response_results.json()[backgroundType]
            for value in pathway_data[:3]:
                dic[value[1]] = [value[2],",".join(value[5]), backgroundType]
        except TypeError:
            continue
    pathway_analysis = []    
    dic_sorted = dict(sorted(dic.items(), key=lambda item: item[1][0]))
    for key, value in dic_sorted.items():
        pathway_analysis.append({"term": key, "overlapping genes": value[1], "database": value[2]})
        # print(pathway_analysis)
    return json.dumps(pathway_analysis[:5])

get_pathway_for_gene_set_doc = {
    "name": "get_pathway_for_gene_set",
    "description": "Given a gene set, return its top-5 biological pathway names.",
    "parameters": {
        "type": "object",
        "properties": {
            "gene_set": {
                "type": "string",
                "description": "A gene set splittd only by comma \",\" (must no whitespace) to search. For example, \"x,y,z\".",
            }
        },
        "required": ["gene_set"],
    },
}


