"""Quick validation of key requirements."""

import json
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_graphs import load_graph
from partitioner import metis_partition
from cypher_parser import parse_match_clause
from query_executor import execute_cypher_query, is_cross_partition


def test_partition_coverage():
    """Test 100% partition coverage for all domains."""
    print("Testing partition coverage...")
    for domain in ["movie", "nba", "company"]:
        graph = load_graph(domain)
        pm = metis_partition(graph, k=4, seed=42)
        unassigned = [n for n in graph.nodes() if n not in pm]
        coverage = len(pm) / graph.number_of_nodes() * 100
        assert len(unassigned) == 0, f"{domain}: {len(unassigned)} unassigned"
        assert coverage == 100.0, f"{domain}: coverage {coverage}%"
        sizes = {}
        for pid in pm.values():
            sizes[pid] = sizes.get(pid, 0) + 1
        print(f"  {domain}: {len(pm)}/{graph.number_of_nodes()} nodes (100%), sizes={sorted(sizes.values())}")


def test_cross_partition_brute_force():
    """Test cross-partition flag matches brute-force on 50 random queries."""
    print("\nTesting cross-partition brute-force (50 sample)...")
    random.seed(42)
    qa_data = json.load(open("d:/Datasets/Cyberbench/test.json"))
    with open("d:/Datasets/Cyberbench/test.json", encoding='utf-8') as f:
        qa_data = json.load(f)
    all_qs = [q for q in qa_data if q['graph'] in ["movie", "nba", "company"]]
    sample = random.sample(all_qs, min(50, len(all_qs)))

    partition_maps = {}
    graphs = {}
    for domain in ["movie", "nba", "company"]:
        g = load_graph(domain)
        graphs[domain] = g
        partition_maps[domain] = metis_partition(g, k=4, seed=42)

    mismatches = 0
    for qa in sample:
        domain = qa['graph']
        touched = execute_cypher_query(graphs[domain], qa['gold_cypher'])
        pipeline_cp = is_cross_partition(touched, partition_maps[domain])
        if touched:
            part_ids = {partition_maps[domain][n] for n in touched if n in partition_maps[domain]}
            brute_cp = len(part_ids) > 1
        else:
            brute_cp = False
        if pipeline_cp != brute_cp:
            mismatches += 1
            print(f"  MISMATCH: {qa['qid']} in {domain}: pipeline={pipeline_cp}, brute={brute_cp}")

    print(f"  All {len(sample)} sampled queries match brute-force check")
    print(f"  Mismatches: {mismatches}")
    assert mismatches == 0, f"Found {mismatches} mismatches"


def test_output_files():
    """Verify output files exist and have correct format."""
    print("\nVerifying output files...")
    for domain in ["movie", "nba", "company"]:
        fp = f"d:/vscode/text_to_gql/data/output/{domain}_augmented.jsonl"
        assert os.path.exists(fp), f"Missing {fp}"
        count = 0
        with open(fp) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line.strip())
                    for field in ["nl_question", "gold_cypher", "domain",
                                  "referenced_labels", "referenced_partitions",
                                  "is_cross_partition", "partition_map_ref"]:
                        assert field in rec, f"Missing {field} in {domain}"
                    assert isinstance(rec['is_cross_partition'], bool)
                    count += 1
        print(f"  {domain}: {count} records validated")
        assert count > 0


def test_partition_maps():
    """Verify partition map files."""
    print("\nVerifying partition map files...")
    for domain in ["movie", "nba", "company"]:
        fp = f"d:/vscode/text_to_gql/data/maps/{domain}_k4.json"
        assert os.path.exists(fp), f"Missing {fp}"
        with open(fp) as f:
            pm = json.load(f)
        assert isinstance(pm, dict)
        assert len(pm) > 0
        print(f"  {domain}: {len(pm)} entries")


if __name__ == "__main__":
    test_partition_coverage()
    test_cross_partition_brute_force()
    test_output_files()
    test_partition_maps()
    print("\n✅ All validation tests passed!")