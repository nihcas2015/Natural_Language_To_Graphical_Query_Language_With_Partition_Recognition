"""
Main pipeline: build augmented CypherBench JSONL dataset with partition info.
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter, defaultdict
from typing import List, Dict, Set, Any

from load_graphs import load_graph, save_partition_map
from cypher_parser import parse_match_clause
from query_executor import execute_cypher_query, get_partition_ids, is_cross_partition

DOMAINS = ["movie", "nba", "company"]
K = 4
DATA_DIR = "d:/vscode/text_to_gql/data"
MAPS_DIR = os.path.join(DATA_DIR, "maps")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
QA_FILE = "d:/Datasets/Cyberbench/test.json"


def load_qa_pairs(domains: List[str]) -> Dict[str, List[dict]]:
    """Load QA pairs for specified domains."""
    all_qa = json.load(open(QA_FILE))
    with open(QA_FILE, encoding='utf-8') as f:
        all_qa = json.load(f)
    result = {}
    for d in domains:
        result[d] = [q for q in all_qa if q['graph'] == d]
        print(f"Loaded {len(result[d])} QA pairs for domain '{d}'")
    return result


def process_domain(domain: str, graph, qa_pairs: List[dict]):
    """Process a single domain: partition, parse queries, build records."""
    print(f"\n=== Processing domain: {domain} ===")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    from partitioner import metis_partition
    partition_map = metis_partition(graph, k=K, seed=42)
    sizes = Counter(partition_map.values())
    print(f"Partition sizes: {dict(sizes)}")

    save_partition_map(domain, partition_map, MAPS_DIR)

    records = []
    cross_count = 0
    for qa in qa_pairs:
        gold_cypher = qa['gold_cypher']
        parsed = parse_match_clause(gold_cypher)
        referenced_labels = parsed['referenced_labels']
        touched_nodes = execute_cypher_query(graph, gold_cypher)
        referenced_partitions = get_partition_ids(touched_nodes, partition_map)
        cp_flag = len(referenced_partitions) > 1
        if cp_flag:
            cross_count += 1
        record = {
            "nl_question": qa['nl_question'],
            "gold_cypher": gold_cypher,
            "domain": domain,
            "referenced_labels": referenced_labels,
            "referenced_partitions": referenced_partitions,
            "is_cross_partition": cp_flag,
            "partition_map_ref": f"{domain}_k4.json",
        }
        records.append(record)

    total = len(records)
    print(f"Domain {domain}: {cross_count}/{total} cross-partition ({100*cross_count/total:.1f}%)")
    return records, partition_map, cross_count, total


def compute_stats(all_records, all_partition_maps, all_graphs):
    """Compute summary statistics for the report."""
    stats = {}
    for domain in DOMAINS:
        records = all_records[domain]
        partition_map = all_partition_maps[domain]
        graph = all_graphs[domain]
        total = len(records)
        cross = sum(1 for r in records if r['is_cross_partition'])
        cross_pct = 100.0 * cross / total if total > 0 else 0.0
        part_sizes = defaultdict(int)
        for pid in partition_map.values():
            part_sizes[pid] += 1
        sizes = sorted(part_sizes.values())
        edge_cut = 0
        for u, v in graph.edges():
            if partition_map.get(u) != partition_map.get(v):
                edge_cut += 1
        edge_cut_ratio = edge_cut / graph.number_of_edges() if graph.number_of_edges() > 0 else 0.0
        stats[domain] = {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "partition_sizes": sizes,
            "edge_cut_ratio": round(edge_cut_ratio, 4),
            "total_queries": total,
            "cross_partition_queries": cross,
            "cross_partition_pct": round(cross_pct, 1),
        }
    return stats


def write_report(stats):
    """Write the markdown report."""
    lines = [
        "# CypherBench Data Preparation Report", "",
        "This report summarizes partitioning and cross-partition analysis",
        "of the CypherBench Movie, NBA, and Company domain graphs (k=4).", "",
        "---", ""
    ]
    for domain in DOMAINS:
        s = stats[domain]
        lines.extend([
            f"## {domain.upper()} Domain", "",
            f"**Graph Statistics:**",
            f"- Nodes: {s['total_nodes']:,}",
            f"- Edges: {s['total_edges']:,}", "",
            f"**Partition Configuration (k=4):**",
            f"- Partition sizes: {s['partition_sizes']}",
            f"- Edge-cut ratio: {s['edge_cut_ratio']:.4f}", "",
            f"**Cross-Partition Analysis:**",
            f"- Total QA pairs: {s['total_queries']}",
            f"- Cross-partition queries: {s['cross_partition_queries']}",
            f"- Cross-partition percentage: {s['cross_partition_pct']:.1f}%", "", "---", ""
        ])
    total_queries = sum(s['total_queries'] for s in stats.values())
    total_cross = sum(s['cross_partition_queries'] for s in stats.values())
    overall_pct = 100.0 * total_cross / total_queries if total_queries > 0 else 0.0
    lines.extend([
        "## Overall Summary", "",
        f"- Total QA pairs: {total_queries}",
        f"- Cross-partition queries: {total_cross}",
        f"- Overall cross-partition %: {overall_pct:.1f}%", "",
        "## Partition Map References", "",
        "Each domain has a partition map file:", ""
    ])
    for d in DOMAINS:
        lines.append(f"- `{d}_k4.json`")
    lines.extend(["", "---", "", "## Methodology", "",
        "1. Graph Loading: CypherBench simplekg JSON format.",
        "2. Partitioning: Custom k-way METIS-like partitioner.",
        "3. Query Parsing: Custom Cypher tokenizer.",
        "4. Query Execution: Gold queries against the graph.",
        "5. Cross-Partition: Physical nodes mapped to partition IDs.", ""])
    with open(os.path.join(OUTPUT_DIR, "REPORT.md"), 'w') as f:
        f.write('\n'.join(lines))
    print(f"Report written to {os.path.join(OUTPUT_DIR, 'REPORT.md')}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    qa_by_domain = load_qa_pairs(DOMAINS)
    all_records = {}
    all_partition_maps = {}
    all_graphs = {}

    for domain in DOMAINS:
        graph = load_graph(domain)
        all_graphs[domain] = graph
        records, partition_map, cross, total = process_domain(domain, graph, qa_by_domain[domain])
        all_records[domain] = records
        all_partition_maps[domain] = partition_map

    for domain in DOMAINS:
        filepath = os.path.join(OUTPUT_DIR, f"{domain}_augmented.jsonl")
        with open(filepath, 'w') as f:
            for rec in all_records[domain]:
                f.write(json.dumps(rec) + '\n')
        print(f"Saved {len(all_records[domain])} records to {filepath}")

    stats = compute_stats(all_records, all_partition_maps, all_graphs)
    with open(os.path.join(OUTPUT_DIR, "summary_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    write_report(stats)
    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()

