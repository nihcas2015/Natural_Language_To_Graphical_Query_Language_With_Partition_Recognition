# CypherBench Data Preparation Report

This report summarizes partitioning and cross-partition analysis
of the CypherBench Movie, NBA, and Company domain graphs (k=4).

---

## MOVIE Domain

**Graph Statistics:**
- Nodes: 459,393
- Edges: 1,811,311

**Partition Configuration (k=4):**
- Partition sizes: [109106, 109106, 109106, 132075]
- Edge-cut ratio: 0.6889

**Cross-Partition Analysis:**
- Total QA pairs: 401
- Cross-partition queries: 389
- Cross-partition percentage: 97.0%

---

## NBA Domain

**Graph Statistics:**
- Nodes: 4,327
- Edges: 16,728

**Partition Configuration (k=4):**
- Partition sizes: [1028, 1028, 1028, 1243]
- Edge-cut ratio: 0.7101

**Cross-Partition Analysis:**
- Total QA pairs: 270
- Cross-partition queries: 254
- Cross-partition percentage: 94.1%

---

## COMPANY Domain

**Graph Statistics:**
- Nodes: 581,306
- Edges: 298,513

**Partition Configuration (k=4):**
- Partition sizes: [138061, 138061, 138061, 167123]
- Edge-cut ratio: 0.6671

**Cross-Partition Analysis:**
- Total QA pairs: 347
- Cross-partition queries: 336
- Cross-partition percentage: 96.8%

---

## Overall Summary

- Total QA pairs: 1018
- Cross-partition queries: 979
- Overall cross-partition %: 96.2%

## Partition Map References

Each domain has a partition map file:

- `movie_k4.json`
- `nba_k4.json`
- `company_k4.json`

---

## Methodology

1. Graph Loading: CypherBench simplekg JSON format.
2. Partitioning: Custom k-way METIS-like partitioner.
3. Query Parsing: Custom Cypher tokenizer.
4. Query Execution: Gold queries against the graph.
5. Cross-Partition: Physical nodes mapped to partition IDs.
