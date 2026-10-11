"""
Load CypherBench property graphs from the simplekg JSON format into NetworkX graphs.

The graph JSON files have the structure:
  {
    "schema": {...},
    "entities": [{"eid": "Movie#Q1000094", "label": "Movie", "name": "...", ...}],
    "relations": [{"rid": "0", "label": "directedBy", "subj_id": "Movie#Q1000094", "obj_id": "Person#Q526222", ...}]
  }

Each entity becomes a node with its eid as the node identifier.
Each relation becomes a directed edge between subj_id and obj_id.
Graph loader for CypherBench simplekg property graphs.
Parses entities and relations into NetworkX DiGraph objects.
"""

import json
import os
import networkx as nx
from typing import Dict, List, Tuple

DATA_DIR = "d:/Datasets/Cyberbench/graphs/simplekg"


def load_graph(domain: str) -> nx.DiGraph:
    """
    Load a single domain graph from the simplekg JSON format.

    Parameters
    ----------
    domain : str
        Domain name (e.g., 'movie', 'nba', 'company')

    Returns
    -------
    nx.DiGraph
        Directed graph with nodes = entities, edges = relations
    """
    filepath = os.path.join(DATA_DIR, f"{domain}_simplekg.json")
    if not os.path.exists(filepath):
        # Try with .json extension explicitly
        filepath = os.path.join(DATA_DIR, f"{domain}_simplekg.json")

    with open(filepath, 'r') as f:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.DiGraph()

    # Add nodes from entities
    for entity in data.get("entities", []):
        eid = entity["eid"]
        G.add_node(eid, **{k: v for k, v in entity.items() if k != "eid"})

    # Add edges from relations
    for rel in data.get("relations", []):
        subj = rel["subj_id"]
        obj = rel["obj_id"]
        label = rel.get("label", "")
        props = {k: v for k, v in rel.items() if k not in ("rid", "label", "subj_id", "obj_id")}
        subj = rel.get("subj_id") or rel.get("head")
        obj = rel.get("obj_id") or rel.get("tail")
        label = rel.get("label", rel.get("type", ""))
        props = {k: v for k, v in rel.items() if k not in ("rid", "label", "subj_id", "obj_id", "head", "tail", "type")}
        if subj in G and obj in G:
            G.add_edge(subj, obj, label=label, **props)

    return G


def load_all_domains(domains: List[str]) -> Dict[str, nx.DiGraph]:
    """Load graphs for all specified domains."""
    graphs = {}
    for domain in domains:
        graphs[domain] = load_graph(domain)
        print(f"Loaded {domain}: {graphs[domain].number_of_nodes()} nodes, {graphs[domain].number_of_edges()} edges")
    return graphs


def get_schema(domain: str) -> dict:
    """Load the schema for a domain."""
    schema_path = os.path.join(
        "d:/Datasets/Cyberbench/graphs/schemas", f"{domain}_schema.json"
    )
    with open(schema_path, 'r') as f:
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_partition_map(domain: str, partition_map: dict, output_dir: str = "d:/vscode/text_to_gql/data/maps"):
    """Save partition map as JSON file."""
    filepath = os.path.join(output_dir, f"{domain}_k4.json")
    with open(filepath, 'w') as f:
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(partition_map, f, indent=2)
    print(f"Saved partition map to {filepath}")


if __name__ == "__main__":
    domains = ["movie", "nba", "company"]
    graphs = {}
    for d in domains:
        graphs[d] = load_graph(d)
        print(f"{d}: {graphs[d].number_of_nodes()} nodes, {graphs[d].number_of_edges()} edges")
        try:
            graphs[d] = load_graph(d)
            print(f"{d}: {graphs[d].number_of_nodes():,} nodes, {graphs[d].number_of_edges():,} edges")
        except Exception as e:
            print(f"Failed {d}: {e}")
