"""Unit tests for the data preparation pipeline."""

import json, os, random, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_graphs import load_graph
from partitioner import metis_partition
from cypher_parser import parse_match_clause
from query_executor import execute_cypher_query, is_cross_partition
from build_dataset import DOMAINS, MAPS_DIR, OUTPUT_DIR, QA_FILE


class TestPartitionCoverage(unittest.TestCase):
    def test_full_coverage_all_domains(self):
        for domain in DOMAINS:
            graph = load_graph(domain)
            pm = metis_partition(graph, k=4, seed=42)
            unassigned = [n for n in graph.nodes() if n not in pm]
            self.assertEqual(len(unassigned), 0, f"{domain}: {len(unassigned)} unassigned")
            coverage = len(pm) / graph.number_of_nodes() * 100
            self.assertEqual(coverage, 100.0, f"{domain}: coverage {coverage}%")
            print(f"  {domain}: {len(pm)}/{graph.number_of_nodes()} nodes (100%)")

    def test_partition_map_files_exist(self):
        for domain in DOMAINS:
            fp = os.path.join(MAPS_DIR, f"{domain}_k4.json")
            self.assertTrue(os.path.exists(fp), f"Missing {fp}")
            with open(fp) as f:
                pm = json.load(f)
            self.assertIsInstance(pm, dict)
            self.assertGreater(len(pm), 0)


class TestCrossPartitionBruteForce(unittest.TestCase):
    def test_cross_partition_sample(self):
        random.seed(42)
        qa_data = json.load(open(QA_FILE))
        with open(QA_FILE, encoding='utf-8') as f:
            qa_data = json.load(f)
        sample_domains = ["movie", "nba", "company"]
        all_samples = [q for d in sample_domains
                        for q in qa_data if q['graph'] == d]
        sample = random.sample(all_samples, min(50, len(all_samples)))
        partition_maps = {}
        graphs = {}
        for domain in sample_domains:
            g = load_graph(domain)
            graphs[domain] = g
            partition_maps[domain] = metis_partition(g, k=4, seed=42)
        mismatches = 0
        for qa in sample:
            domain = qa['graph']
            touched = execute_cypher_query(graphs[domain], qa['gold_cypher'])
            pipeline_cp = is_cross_partition(touched, partition_maps[domain])
            if touched:
                part_ids = {partition_maps[domain][n] for n in touched
                           if n in partition_maps[domain]}
                brute_cp = len(part_ids) > 1
            else:
                brute_cp = False
            self.assertEqual(pipeline_cp, brute_cp, f"Mismatch: {qa['qid']} in {domain}")
        print(f"\n  All {len(sample)} sampled queries match brute-force check")

    def test_cross_partition_stats_consistency(self):
        qa_data = json.load(open(QA_FILE))
        with open(QA_FILE, encoding='utf-8') as f:
            qa_data = json.load(f)
        partition_maps = {}
        for domain in DOMAINS:
            g = load_graph(domain)
            partition_maps[domain] = metis_partition(g, k=4, seed=42)
        for domain in DOMAINS:
            domain_qs = [q for q in qa_data if q['graph'] == domain]
            cross = sum(1 for q in domain_qs
                       if is_cross_partition(
                           execute_cypher_query(load_graph(domain), q['gold_cypher']),
                           partition_maps[domain]))
            pct = 100.0 * cross / len(domain_qs) if domain_qs else 0
            self.assertGreaterEqual(cross, 0)
            self.assertLessEqual(cross, len(domain_qs))
            print(f"  {domain}: {cross}/{len(domain_qs)} cross-partition ({pct:.1f}%)")


class TestCypherParser(unittest.TestCase):
    def test_parse_labels(self):
        query = "MATCH (n:Movie)-[r:directedBy]->(m:Person) RETURN n.name"
        parsed = parse_match_clause(query)
        self.assertIn("Movie", parsed['referenced_labels'])
        self.assertIn("Person", parsed['referenced_labels'])

    def test_parse_no_label(self):
        query = "MATCH (n) WHERE n.name = 'foo' RETURN n"
        parsed = parse_match_clause(query)
        self.assertEqual(parsed['referenced_labels'], [])


class TestAugmentedDataset(unittest.TestCase):
    def test_jsonl_files_exist(self):
        for domain in DOMAINS:
            fp = os.path.join(OUTPUT_DIR, f"{domain}_augmented.jsonl")
            self.assertTrue(os.path.exists(fp), f"Missing {fp}")
            count = 0
            with open(fp) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line.strip())
                        for field in ["nl_question", "gold_cypher", "domain",
                                      "referenced_labels", "referenced_partitions",
                                      "is_cross_partition", "partition_map_ref"]:
                            self.assertIn(field, rec)
                        self.assertIsInstance(rec['is_cross_partition'], bool)
                        count += 1
            print(f"  {domain}: {count} records validated")
            self.assertGreater(count, 0)

    def test_partition_map_ref_consistency(self):
        for domain in DOMAINS:
            expected = f"{domain}_k4.json"
            fp = os.path.join(OUTPUT_DIR, f"{domain}_augmented.jsonl")
            with open(fp) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line.strip())
                        self.assertEqual(rec['partition_map_ref'], expected)


if __name__ == "__main__":
    if not os.path.exists(os.path.join(OUTPUT_DIR, "movie_augmented.jsonl")):
        print("Running pipeline first...")
        from build_dataset import main
        main()
    print("\nRunning tests...\n")
    unittest.main(verbosity=2)