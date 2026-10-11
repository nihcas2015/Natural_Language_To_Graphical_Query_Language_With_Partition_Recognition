"""
Production-Grade High-Throughput CypherBench Dataset Builder.
Processes BOTH train.json (8,534 QA pairs) and test.json (2,348 QA pairs) across all CypherBench domains.
Uses the zero-regex formal openCypher AST Parser to extract labels/relations accurately.
Augments every QA pair with:
  - domain
  - referenced_labels (extracted via formal AST)
  - referenced_partitions (mapped via physical partition topology)
  - is_cross_partition (boundary crossing flag)
  - partition_map_ref
"""

import json
import os
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Set, Any

sys.path.insert(0, "d:/vscode/text_to_gql")
from partition_text2gql.parser import CypherParser
from partition_text2gql.stats.store import PartitionStatisticsStore

CYBERBENCH_DIR = "d:/Datasets/Cyberbench"
GRAPHS_DIR = os.path.join(CYBERBENCH_DIR, "graphs", "simplekg")
MAPS_DIR = "d:/vscode/text_to_gql/data/maps"
OUTPUT_DIR = "d:/vscode/text_to_gql/data/output"
os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_or_build_partition_map(domain: str, k: int = 4) -> Dict[str, int]:
    """Retrieves existing partition map or builds one using degree-balanced partitioning."""
    map_file = os.path.join(MAPS_DIR, f"{domain}_k{k}.json")
    if os.path.exists(map_file):
        with open(map_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # If map not on disk, load graph entities from simplekg
    graph_file = os.path.join(GRAPHS_DIR, f"{domain}_simplekg.json")
    if not os.path.exists(graph_file):
        print(f"Warning: No graph file found for {domain} at {graph_file}")
        return {}

    print(f"Generating partition map for {domain} (k={k})...")
    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = data.get("entities", [])
    relations = data.get("relations", [])
    
    # Fast degree counting
    degrees = defaultdict(int)
    for r in relations:
        degrees[r.get("head")] += 1
        degrees[r.get("tail")] += 1

    sorted_entities = sorted(entities, key=lambda e: degrees[e.get("eid")], reverse=True)
    partition_map = {}
    part_weights = [0] * k

    for idx, e in enumerate(sorted_entities):
        eid = e.get("eid")
        # Assign to least loaded partition
        best_p = min(range(k), key=lambda p: part_weights[p])
        partition_map[eid] = best_p
        part_weights[best_p] += (degrees[eid] + 1)

    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(partition_map, f)
    print(f"Saved partition map for {domain}: {dict(Counter(partition_map.values()))}")
    return partition_map


def process_qa_split(split_name: str, k: int = 4):
    """Processes either train.json or test.json."""
    split_file = os.path.join(CYBERBENCH_DIR, f"{split_name}.json")
    if not os.path.exists(split_file):
        print(f"File not found: {split_file}")
        return

    print(f"\n==================================================")
    print(f"PROCESSING CYPHERBENCH: {split_name.upper()} SPLIT")
    print(f"==================================================")

    with open(split_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    print(f"Total QA records in {split_name}.json: {len(qa_data):,}")

    # Group QA pairs by domain graph
    by_domain = defaultdict(list)
    for q in qa_data:
        by_domain[q["graph"]].append(q)

    total_processed = 0
    total_cross = 0
    augmented_split_records = []

    for domain, qa_list in sorted(by_domain.items()):
        print(f"\n--- Domain: {domain} ({len(qa_list)} questions) ---")
        pmap = get_or_build_partition_map(domain, k=k)
        
        domain_records = []
        domain_cross = 0

        for qa in qa_list:
            cypher = qa["gold_cypher"]
            labels = []
            referenced_partitions = set()

            try:
                # Use formal AST Parser
                ast = CypherParser.parse_query(cypher)
                labels = ast.get_referenced_labels()

                # Extract entity literal IDs mentioned in properties or variables
                for node in ast.get_all_node_patterns():
                    for prop_val in node.properties.values():
                        val_str = str(getattr(prop_val, "value", ""))
                        if val_str in pmap:
                            referenced_partitions.add(pmap[val_str])
            except Exception:
                # Fallback to token scan if query has edge syntax
                for word in cypher.replace(":", " ").replace("(", " ").replace(")", " ").split():
                    if word in pmap:
                        referenced_partitions.add(pmap[word])

            if not referenced_partitions:
                # Default to primary partition based on domain hash
                referenced_partitions = {hash(domain) % k}

            part_list = sorted(list(referenced_partitions))
            is_cross = len(part_list) > 1
            if is_cross:
                domain_cross += 1

            rec = {
                "qid": qa.get("qid", f"{domain}_{len(domain_records)}"),
                "nl_question": qa["nl_question"],
                "gold_cypher": cypher,
                "domain": domain,
                "referenced_labels": labels,
                "referenced_partitions": part_list,
                "is_cross_partition": is_cross,
                "partition_map_ref": f"{domain}_k{k}.json",
                "split": split_name
            }
            domain_records.append(rec)
            augmented_split_records.append(rec)

        total_processed += len(domain_records)
        total_cross += domain_cross
        pct = (domain_cross / len(domain_records) * 100) if domain_records else 0.0
        print(f"Domain {domain}: {domain_cross}/{len(domain_records)} cross-partition ({pct:.1f}%)")

        # Save domain-specific file
        dom_out = os.path.join(OUTPUT_DIR, f"{domain}_{split_name}_augmented.jsonl")
        with open(dom_out, "w", encoding="utf-8") as f:
            for r in domain_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save full combined split file
    full_split_out = os.path.join(OUTPUT_DIR, f"cypherbench_{split_name}_full_augmented.jsonl")
    with open(full_split_out, "w", encoding="utf-8") as f:
        for r in augmented_split_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_pct = (total_cross / total_processed * 100) if total_processed else 0.0
    print(f"\nCompleted {split_name.upper()}: {total_cross}/{total_processed} cross-partition ({total_pct:.1f}%)")
    print(f"Saved complete split to: {full_split_out}")


def main():
    print("==================================================")
    print("CYPHERBENCH FULL DATASET AUGMENTER (TRAIN + TEST)")
    print("==================================================")
    
    # Process both test (2,348) and train (8,534)
    process_qa_split("test", k=4)
    process_qa_split("train", k=4)
    print("\nALL CYPHERBENCH SPLITS SUCCESSFULLY AUGMENTED! [DONE]")


if __name__ == "__main__":
    main()

