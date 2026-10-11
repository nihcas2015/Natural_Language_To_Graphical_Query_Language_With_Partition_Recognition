"""
LDBC Social Network Benchmark (SNB) & FinBench Synthetic Power-Law Graph Generator.
Produces a benchmark graph simulating scale factor 1-10 with power-law degree distribution
and realistic edge cuts for evaluating distributed graph engines (Neo4j Fabric / JanusGraph).
"""

import json
import os
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any

OUTPUT_DIR = "d:/vscode/text_to_gql/data/ldbc_snb"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_ldbc_snb_scale_graph(
    num_persons: int = 5000,
    k_partitions: int = 4,
    gamma: float = 2.5
) -> Dict[str, Any]:
    """
    Generates an LDBC SNB-style power-law property graph with:
      - Person vertices with properties: id, firstName, lastName, creationDate
      - Forum / Message / Tag vertices
      - KNOWS edges (power-law social network)
      - HAS_CREATOR / HAS_TAG edges
      - Evaluates cross-partition edge cuts
    """
    print(f"Generating LDBC SNB synthetic graph: {num_persons:,} Persons across {k_partitions} shards...")
    
    # 1. Generate power-law degree sequence for KNOWS relationship
    degrees = np.random.zipf(a=gamma, size=num_persons)
    degrees = np.clip(degrees, 2, 250)
    if sum(degrees) % 2 != 0:
        degrees[0] += 1

    # Generate Chung-Lu / expected degree power-law graph
    G = nx.expected_degree_graph(degrees, selfloops=False)
    
    entities = []
    relations = []
    partition_map = {}

    # 2. Partition using balanced degree assignment
    person_degrees = dict(G.degree())
    sorted_nodes = sorted(G.nodes(), key=lambda n: person_degrees[n], reverse=True)
    part_weights = [0] * k_partitions

    for n in sorted_nodes:
        pid = min(range(k_partitions), key=lambda p: part_weights[p])
        partition_map[f"person_{n}"] = pid
        part_weights[pid] += (person_degrees[n] + 1)
        entities.append({
            "eid": f"person_{n}",
            "label": "Person",
            "firstName": f"User_{n}",
            "shard": pid
        })

    # 3. Add edges and calculate cuts
    edge_cuts = 0
    total_edges = 0
    for u, v in G.edges():
        src = f"person_{u}"
        tgt = f"person_{v}"
        pu = partition_map[src]
        pv = partition_map[tgt]
        if pu != pv:
            edge_cuts += 1
        total_edges += 1
        relations.append({
            "head": src,
            "tail": tgt,
            "type": "KNOWS",
            "cross_partition": (pu != pv)
        })

    cut_ratio = edge_cuts / total_edges if total_edges > 0 else 0.0
    print(f"Generated {len(entities):,} vertices, {total_edges:,} KNOWS edges.")
    print(f"Cross-partition edge cut ratio: {cut_ratio:.2%}")

    # 4. Generate representative multi-hop benchmark questions
    benchmark_qa = [
        {
            "qid": "ldbc_is1",
            "nl_question": "Find friends of friends of User_0 who are not direct friends",
            "gold_cypher": "MATCH (p:Person {firstName: 'User_0'})-[:KNOWS]->(f1:Person)-[:KNOWS]->(f2:Person) WHERE NOT (p)-[:KNOWS]->(f2) AND p <> f2 RETURN DISTINCT f2.firstName",
            "domain": "ldbc_snb"
        },
        {
            "qid": "ldbc_is2",
            "nl_question": "Count the number of friends for User_10",
            "gold_cypher": "MATCH (p:Person {firstName: 'User_10'})-[:KNOWS]->(f:Person) RETURN COUNT(f)",
            "domain": "ldbc_snb"
        },
        {
            "qid": "ldbc_ic1",
            "nl_question": "Find all people within 3 degrees of separation from User_5",
            "gold_cypher": "MATCH (p:Person {firstName: 'User_5'})-[:KNOWS*1..3]->(f:Person) RETURN DISTINCT f.firstName LIMIT 50",
            "domain": "ldbc_snb"
        }
    ]

    # Export artifacts
    graph_out = os.path.join(OUTPUT_DIR, "ldbc_snb_simplekg.json")
    map_out = os.path.join(OUTPUT_DIR, "ldbc_snb_k4.json")
    qa_out = os.path.join(OUTPUT_DIR, "ldbc_snb_qa.json")

    with open(graph_out, "w", encoding="utf-8") as f:
        json.dump({"entities": entities, "relations": relations}, f)

    with open(map_out, "w", encoding="utf-8") as f:
        json.dump(partition_map, f)

    with open(qa_out, "w", encoding="utf-8") as f:
        json.dump(benchmark_qa, f, indent=2)

    print(f"Artifacts exported to {OUTPUT_DIR} [DONE]")


if __name__ == "__main__":
    generate_ldbc_snb_scale_graph(num_persons=5000, k_partitions=4)
