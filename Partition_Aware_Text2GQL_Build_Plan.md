# Partition-Aware Text-to-GQL — Build Plan, Prompts & Dataset Guide

**Purpose of this document:** a working playbook you can execute against. It splits your architecture (from your diagrams + Review-1 deck + research plan PDF) into (A) genuine NLP/ML components you train on Kaggle, and (B) systems-engineering components you hand to a coding agent (Claude Code, etc.). Each component has: what it is, which dataset(s) back it, and a ready-to-paste prompt.

---

## 0. Component Map (from your architecture diagram)

| # | Component | Type | Where it's built |
|---|---|---|---|
| 1 | Intent Classification | NLP model | Kaggle |
| 2 | Named Entity Recognition (entities/attributes/values) | NLP model | Kaggle |
| 3 | Schema Linking (NL → labels/edge types/properties + **partition ID**) | NLP model (retrieval) | Kaggle |
| 4 | Semantic Parser / Text-to-GQL generator | NLP model (LLM fine-tune) | Kaggle |
| 5 | Query Validator (syntax / schema conformance / safety) | Systems (grammar + rules), not trainable | Coding agent |
| 6 | Partition-Aware Query Planner (identification + decomposition) | Systems (AST algorithm) — the paper's novel contribution | Coding agent |
| 7 | Distributed Query Execution Engine | Systems (DB infra + dispatch) | Coding agent |
| 8 | Result Aggregator / Response Formatter | Systems | Coding agent |
| 9 | Cost estimator (cross-partition cost prediction, used to rank candidate queries) | Small ML model (regression/ranking) — optional but strengthens novelty | Kaggle |
| 10 | Query Validator & Correction feedback loop | Systems + prompts to the LLM | Coding agent |
| 11 | Evaluation harness (EX, PSJS, latency, network-hop reduction) | Systems/scripts | Coding agent |

**Critical thing to internalize before you start:** none of the public datasets have partition labels. That gap is exactly your paper's contribution, per your own literature review. So Phase 0 is not optional — you must synthetically partition a real graph (via METIS/Fennel) and propagate partition IDs down into the schema-linking and query-generation training data. Every Kaggle notebook below assumes this augmented dataset exists.

---

## 1. Datasets — what to use for what

| Dataset | Source | Use it for | Notes |
|---|---|---|---|
| **CypherBench** (Wikidata subgraphs, 11 domains, ~7.8M nodes / 35M edges, 10K+ NL↔Cypher pairs) | `github.com/megagonlabs/cypherbench`, HF `megagonlabs/cypherbench` | Primary training/eval set for schema linking + Text-to-Cypher generation. Also your base graph for partitioning experiments (it's the same dataset your PDF recommends METIS on). | Use 2–3 domains (e.g. Movies, NBA, Companies) at full scale for training; keep 1–2 held out for cross-domain generalization eval. |
| **Neo4j Text2Cypher (2024)** | HF `neo4j/text2cypher-2024v1` | Secondary fine-tuning corpus for the generation model (adds enterprise-style schema diversity, improves robustness beyond Wikidata phrasing). | ~44K examples across many schemas — good for teaching schema-agnostic generation before you specialize on your partitioned schema. |
| **LDBC Social Network Benchmark (SNB)** | `ldbcouncil.org`, `github.com/ldbc/ldbc_snb_datagen_spark` | Distributed-systems side: generate a scalable synthetic graph (start at Scale Factor 1–10) to actually partition and run your engine against. Good power-law structure for realistic edge-cut testing. | Use this as your "systems benchmark" graph, separate from CypherBench which is your "NLP benchmark" graph — see §1.1 on why you need both. |
| **FinBench** | `github.com/ldbc/ldbc_finbench_datagen` | Alternative/second partitioning benchmark (financial domain, hash-shardable on Account ID) — good for a second dataset row in your results table to show generality. | Optional; include if you have time for a second domain in Phase 4. |
| **Multilingual Text2Cypher** | HF, per your PDF citation 5 | Optional — only if you want a "generalization" ablation. Not required for the core contribution. | Skip unless you have spare time in Month 5–6. |

### 1.1 Why you need two graph tracks
- **NLP track (CypherBench / Neo4j-T2C):** used to train intent classification, NER, schema linking, and the generation model — because these need NL↔query pairs.
- **Systems track (LDBC SNB / FinBench, partitioned via METIS/Fennel):** used to actually stand up a partitioned Neo4j Fabric / JanusGraph cluster and measure latency/network-hop reduction — because these need scale and known partitioning behavior, not NL pairs.

Your novel artifact is the bridge: you run METIS on the **CypherBench** graph itself (not just LDBC) so that your schema-linking training data can carry real partition-ID labels tied to real NL questions. LDBC/FinBench are there purely to prove the systems side scales.

---

## 2. Phase 0 (do this before any Kaggle training): Partition-Augmented Dataset Construction

This has to happen once, produces the labeled data every subsequent notebook consumes, and is itself a defensible methodological step for your paper (cite it as "partition-label synthesis" in your Methods section).

**Coding-agent prompt — run this locally or in a CPU Kaggle notebook (no GPU needed):**

```
You are building a data-preparation pipeline for a research project. Do the following,
producing a well-documented, reproducible Python script (not just a notebook cell dump):

1. Load the CypherBench dataset (megagonlabs/cypherbench) for these domains: Movies, NBA,
   Companies. Parse each domain's property graph (nodes, edges, labels, properties) into a
   NetworkX or igraph graph object.

2. Partition each domain graph into k=4 partitions using METIS (via the `pymetis` Python
   binding). Balance vertex counts across partitions and minimize edge-cut. Store the
   output as a partition map: {node_id: partition_id}.

3. For each NL question in the CypherBench QA pairs, parse the gold Cypher query's AST
   (use the `cypher-linter` or `antlr4-opencypher` grammar, or a minimal custom Cypher
   tokenizer if a full parser is unavailable) to extract:
     - all node labels and their variable bindings referenced in the MATCH clause
     - the partition_id(s) each referenced node instance falls into (join the gold
       execution's actual node IDs against the partition map — you will need to execute
       gold queries against the graph to resolve which physical nodes they touch)
     - a boolean "is_cross_partition" flag: true if resolved nodes span >1 partition

4. Emit an augmented JSONL dataset with one record per QA pair, adding fields:
   {
     "nl_question": ..., "gold_cypher": ..., "domain": ...,
     "referenced_labels": [...], "referenced_partitions": [...],
     "is_cross_partition": true/false,
     "partition_map_ref": "movies_k4.json"
   }

5. Save partition maps per domain as separate JSON files. Save summary statistics
   (edge-cut ratio, % cross-partition questions per domain, partition size balance) as a
   markdown report — I will need these numbers for the paper's dataset section.

6. Write unit tests validating that partition assignment coverage is 100% (every node has
   an assigned partition) and that the cross-partition flag matches a brute-force check on
   a random sample of 50 queries.

Output: a `data_prep/` directory with scripts, a `data/` directory with the augmented
JSONL + partition maps, and `REPORT.md` with the statistics.
```

Once this exists, every notebook below reads from `data/*.jsonl` + `data/*_partition_map.json`.

---

## 3. Kaggle Notebooks (NLP model training)

General Kaggle setup note for all notebooks below: use a **P100 or T4x2** accelerator, `transformers`, `peft`, `bitsandbytes`, `datasets`, `sentence-transformers`. Pin versions in a requirements cell for reproducibility (a reviewer will ask).

### Notebook 1 — Intent Classification + NER (joint model)

**What it does:** classifies query intent (lookup / path / aggregation / multi-hop) and extracts entity mentions + attribute/value spans from the NL question. This feeds the schema linker.

```
Build a Kaggle notebook that trains a joint intent-classification + NER model for a
Text-to-GQL pipeline.

Data: data/{domain}_augmented.jsonl (from Phase 0). Derive:
  - intent label per question: bucket into {lookup, path_traversal, aggregation,
    multi_hop_join} using rule-based heuristics over the gold Cypher AST (e.g. presence
    of COUNT/AVG/SUM -> aggregation; path length >=3 hops -> multi_hop_join; single
    MATCH+WHERE -> lookup; variable-length pattern -> path_traversal). Save these derived
    labels back into the JSONL for reuse.
  - NER labels: use weak supervision — align entity/value literals in the gold Cypher
    query back to spans in the NL question via fuzzy string matching, then produce
    BIO-tagged tokens (ENTITY, ATTRIBUTE, VALUE tags).

Model: fine-tune `distilbert-base-uncased` (or `bert-base-uncased` if compute allows)
with two heads: a sequence classification head (intent) and a token classification head
(NER), trained jointly with summed cross-entropy losses, using HuggingFace `transformers`
Trainer with a custom multi-task Dataset/collator.

Split: 80/10/10 train/val/test, stratified by domain and by intent label. Hold out one
full domain (e.g. Companies) as a zero-shot generalization test set — report accuracy on
it separately.

Report: intent accuracy/F1 per class (confusion matrix), NER entity-level F1 (seqeval),
and the zero-shot domain transfer numbers, as this last number matters for the paper's
"generalization" claim. Save the model + tokenizer to `/kaggle/working/intent_ner_model/`
and log all metrics to a `metrics.json` I can paste into a LaTeX table.

Keep the code modular: `data.py`, `model.py`, `train.py`, `evaluate.py`, not one
monolithic notebook cell.
```

### Notebook 2 — Partition-Aware Schema Linking Encoder

**What it does:** the actual novel NLP contribution. A bi-encoder that, given an NL question, retrieves the correct graph labels/properties/edge-types **and** their owning partition(s).

```
Build a Kaggle notebook that trains a partition-aware schema-linking encoder as a
bi-encoder retrieval model.

Data: data/{domain}_augmented.jsonl + partition maps from Phase 0.

Construction of training pairs:
  - Positive pairs: (NL question, schema-element-description) for every label/property/
    edge-type referenced in the gold query, where schema-element-description is a short
    templated string like "Movie.release_year (property, partition 2)" or
    "ACTED_IN (edge type, spans partitions 1,3)".
  - Hard negatives: schema elements from the SAME domain that are NOT referenced in the
    gold query, prioritizing ones with high lexical overlap with the question (to force
    the model to learn semantic + partition grounding, not just keyword matching).

Model: fine-tune a `sentence-transformers` bi-encoder (start from
`sentence-transformers/all-mpnet-base-v2`) using MultipleNegativesRankingLoss over
(question, positive schema-element) pairs with in-batch + hard negatives.

Indexing: after training, build a FAISS (or simple cosine) index over ALL schema element
descriptions (including their partition metadata) per domain, so that at inference time
a question embeds once and retrieves top-k schema elements + their partitions in one
lookup.

Evaluation metrics (report all of these, they map directly to paper claims):
  - Recall@k (k=1,3,5) for correct schema element retrieval
  - Partition-attribution accuracy: of the retrieved schema elements, what fraction have
    the CORRECT partition ID attached (this is your core novelty metric — no existing
    Text2Cypher paper reports this, so define it clearly in the notebook's markdown docs)
  - Cross-partition detection accuracy: does retrieval correctly flag is_cross_partition
    when top-k retrieved elements span >1 partition, versus the ground truth flag

Save the fine-tuned encoder + FAISS index artifacts to /kaggle/working/schema_linker/,
and a metrics.json with all of the above broken down by domain.
```

### Notebook 3 — Grammar-Constrained Text-to-Cypher Generation Model (core generation component)

**What it does:** the LLM that actually writes the Cypher query, conditioned on the schema-linking output (including partition metadata), with grammar constraints so output is always syntactically valid.

```
Build a Kaggle notebook that fine-tunes a small language model for schema-and-partition-
conditioned Text-to-Cypher generation using LoRA.

Base model: Qwen2.5-Coder-7B-Instruct (load in 4-bit via bitsandbytes to fit a single
T4/P100; if that OOMs, fall back to Qwen2.5-Coder-1.5B-Instruct and note the substitution
in metrics.json for reproducibility).

Data construction: for each example in data/{domain}_augmented.jsonl, build a prompt of
the form:
  "Schema context: {top-5 schema elements from Notebook 2's retrieval output, each
   annotated with partition ID}
   Question: {nl_question}
   Generate the openCypher query. If the traversal spans multiple partitions, annotate
   which MATCH clause segments belong to which partition using a `// partition:N` comment
   above each segment."
  Target: {gold_cypher, with partition-segment comments injected using the ground-truth
   partition map from Phase 0}

  This partition-annotated target format is what lets your downstream AST rewriter (built
  by the coding agent, see §4) consume the LLM's output without re-deriving partition
  membership from scratch — make sure the annotation format is simple and consistent
  (e.g. a single-line comment convention) so it's trivial to regex-parse downstream.

Training: LoRA fine-tune (r=16, alpha=32, target_modules=["q_proj","v_proj"]) for 3
epochs, using the `peft` + `transformers` Trainer, with a validation split held out per
domain plus the fully-held-out zero-shot domain from Notebook 1's split for consistency.

Grammar constraining at INFERENCE (implement in this same notebook's eval section, not
just training): after generation, validate each candidate against an openCypher ANTLR4
grammar (use the `antlr4-tools` generated Python parser, or `cypher-linter` if the ANTLR4
grammar setup is too heavy for Kaggle) and reject/resample invalid candidates using
self-consistency sampling across 3 temperatures (0.2, 0.5, 0.8), keeping the
majority-consistent valid candidate.

Metrics to report (needed for your paper's core results table):
  - Execution Accuracy (EX): re-execute the generated query against the actual graph
    (load into an in-memory NetworkX-based Cypher-subset interpreter if you don't have a
    live Neo4j instance on Kaggle, or export queries and evaluate them later against the
    coding agent's live cluster — note this dependency clearly in the notebook)
  - Property Set Jaccard Similarity (PSJS) between predicted and gold result sets
  - Grammar validity rate before vs after constrained resampling
  - Partition-annotation accuracy: % of generated partition comments matching ground truth
  - Compare against a **non-partition-aware baseline**: same model/data but WITHOUT
    partition metadata in the prompt and WITHOUT partition-comment targets — this
    ablation is essential, it's literally your paper's headline comparison (Path A vs
    Path B from your architecture diagram).

Save LoRA adapter weights, the merged model (if VRAM allows), and metrics.json with
per-domain and aggregate numbers, plus the ablation comparison table.
```

### Notebook 4 — Cross-Partition Cost Estimator (optional but strengthens novelty)

**What it does:** given a candidate query's AST + partition annotations, predicts expected cross-partition network cost, used to re-rank multiple candidate queries at test time (your PDF's "Cost-Guided Test-Time Sampling" idea).

```
Build a Kaggle notebook that trains a lightweight cost estimator for candidate Cypher
queries.

Data: for each query in data/{domain}_augmented.jsonl, extract features from its AST +
partition annotations:
  - number of MATCH clause hops
  - number of distinct partitions touched
  - number of cross-partition edge traversals (edges whose endpoints are in different
    partitions, from the partition map)
  - presence of aggregation/ORDER BY (forces full result materialization before shipping)
  - estimated result cardinality proxy (node label frequency count in that domain)

Label: actual cross-partition network hop count, computed by simulating execution of the
gold query hop-by-hop against the partition map (a hop counts as a network transfer if
consecutive traversal steps cross a partition boundary).

Model: train a gradient-boosted regressor (LightGBM or XGBoost — no GPU needed, CPU
notebook is fine) predicting log(1 + hop_count) from the above features. This is
intentionally simple and interpretable (a reviewer will like that it's not a black box
buried inside the LLM).

Evaluation: report MAE / RMSE on held-out queries, plus a ranking metric — given multiple
candidate queries for the same question (from Notebook 3's self-consistency sampling),
does the cost estimator correctly rank the partition-local candidate above the
cross-partition one? Report this as "correct re-ranking rate."

Save the trained model (joblib) and a feature-importance plot (for the paper's discussion
section) to /kaggle/working/cost_estimator/.
```

---

## 4. Coding Agent Prompts (systems engineering — hand these to Claude Code)

These assume the Kaggle-trained artifacts (schema linker index, generation model/adapter, cost estimator) are downloaded locally and available as files.

### Task A — Repository scaffolding + partitioned database infrastructure

```
Scaffold a Python repository for a research prototype called "partition-aware-text2gql".

Structure:
  partition_aware_text2gql/
    infra/           # docker-compose for the DB cluster
    schema_linking/  # loads the FAISS index + sentence-transformer encoder from Kaggle
    generation/      # loads the LoRA-adapted Qwen model, does grammar-constrained decode
    rewriter/        # AST-based partition-aware query rewriter (build in Task B)
    execution/       # dispatches subqueries to shards, aggregates results (Task C)
    validation/      # syntax/schema/safety validator + correction loop (Task D)
    evaluation/       # EX/PSJS/latency harness (Task E)
    api/             # FastAPI service wiring the whole pipeline end to end
    tests/
    README.md

Set up docker-compose.yml for a JanusGraph + Apache Cassandra cluster (4 Cassandra nodes
simulating 4 shards to match the k=4 partitioning used in the Kaggle data prep), following
the setup notes in my research plan (Murmur3Partitioner, vertex-centric indices, explicit
storage partition keys on primary entity labels). Include a second docker-compose profile
for Neo4j Fabric (1 coordinator + 4 shard databases) as an alternative backend, since the
paper should report results on at least one of these two — default to JanusGraph+Cassandra
unless I say otherwise.

Include a data-loader script that ingests the CypherBench domain graphs (per Phase 0's
partition maps) into the cluster such that each partition's nodes land on the intended
physical shard — this is critical, the whole experiment is invalid if partition
assignment doesn't match the labels the NLP models were trained on.

Set up pytest with a smoke test that connects to the cluster and runs a trivial
cross-partition query, confirming multi-shard dispatch works before any pipeline code is
built on top.
```

### Task B — Partition-Aware AST Query Rewriter (the paper's core systems contribution)

```
Implement the partition-aware query rewriting engine in rewriter/.

Input: a raw openCypher query string with `// partition:N` annotation comments (produced
by the fine-tuned generation model), plus the partition map JSON for the active domain.

Steps to implement:
1. Parse the query into an AST using an openCypher grammar (use `antlr4-opencypher` or
   `neo4j/cypher-dsl`-equivalent Python tooling — pick whichever has better Python
   support and document the choice).
2. Build an AST visitor that walks MATCH clause patterns and, for each node/relationship
   pattern, resolves its partition assignment either from (a) the `// partition:N`
   annotation if present and trustworthy, or (b) a fallback lookup against the partition
   map using the pattern's label + any bound literal filters.
3. Decompose the AST into per-partition subquery fragments:
   - Push WHERE predicates down to the partition that owns the filtered variable.
   - Reorder multi-hop traversal steps to keep as many consecutive hops within a single
     partition as possible before crossing a boundary (greedy reordering — implement and
     document the heuristic clearly, this is a defensible novel algorithm for the paper).
   - Insert an explicit boundary-join operator only at points where crossing partitions
     is unavoidable, carrying forward only the minimal join-key columns (not full node
     properties) to minimize network payload — this corresponds to the "Result
     Aggregator" reducing cross-partition join & merge cost in my architecture diagram.
4. Emit a structured execution plan: a list of {partition_id, subquery_string} plus a
   final aggregation/join specification, serialized as JSON so Task C can consume it
   without re-parsing Cypher.

Also implement the UNOPTIMIZED baseline path (Path A from my diagram): dispatch the raw,
unmodified query to a single coordinator and let JanusGraph/Fabric's own engine handle it
naively, with no partition-aware rewriting. I need both paths runnable side by side so
the evaluation harness in Task E can produce the Path A vs Path B comparison table.

Write extensive unit tests with hand-crafted Cypher queries covering: single-partition
queries (no rewrite needed), 2-hop cross-partition queries, 4+ hop queries touching 3+
partitions, and queries with aggregation (COUNT/AVG) that must NOT be split incorrectly
across partitions (aggregation semantics must remain correct after decomposition — add a
correctness test that compares decomposed-execution results against non-decomposed
execution on a small in-memory graph to prove semantic equivalence).
```

### Task C — Distributed Execution Dispatcher + Result Aggregator

```
Implement execution/ to consume the JSON execution plan from Task B's rewriter and
actually run it against the JanusGraph/Cassandra (or Neo4j Fabric) cluster from Task A.

Requirements:
  - Dispatch each {partition_id, subquery_string} fragment to its owning shard in
    parallel (asyncio or a thread pool — the cluster clients are likely blocking, so use
    a thread pool with one connection per shard).
  - Collect per-shard result sets and perform the final boundary join/aggregation exactly
    as specified in the execution plan.
  - Implement execution sandboxing: a hard 5-second timeout per subquery (per my research
    plan's risk-mitigation section), cancelling and logging a timeout error rather than
    hanging.
  - Instrument every dispatch with timing (per-shard latency, total end-to-end latency,
    number of network hops actually incurred, bytes transferred at each boundary join) —
    these are the exact metrics needed for the Cost Comparison & Analytics box in my
    architecture diagram, so log them in a structured format (JSON lines) the evaluation
    harness can aggregate later.
  - Also implement the naive dispatcher for Path A (send the whole unmodified query to a
    single coordinator/shard and let the underlying engine's own distributed execution
    handle it, still logging the same latency/hop metrics for comparison).

Return both the query result and a metrics object from every call, so the API layer
(Task F) and evaluation harness (Task E) share one consistent metrics schema.
```

### Task D — Query Validator + Correction Feedback Loop

```
Implement validation/ implementing the three validator boxes from my architecture
diagram: Syntax Validator, Schema Conformance Checker, Access/Safety Checker.

- Syntax Validator: reuse the ANTLR4 grammar parser from Task B; if parsing fails, return
  a structured error with the offending token/position.
- Schema Conformance Checker: verify every label/property/edge-type referenced in the
  parsed AST actually exists in the domain's schema (load schema metadata from the same
  source Notebook 2 used), catching hallucinated labels/properties from the generation
  model.
- Access/Safety Checker: reject queries containing write operations (CREATE, DELETE,
  SET, MERGE, REMOVE) since this system is read-only by design; also reject queries with
  unbounded variable-length path patterns (e.g. `[*]` with no upper bound) since my
  research plan flags these as a known failure mode causing exponential memory blowup —
  reject with a clear error rather than executing.

If validation fails, implement the "retry with correction" feedback loop shown in my
diagram: construct a corrective prompt containing the original NL question, the failed
query, and the specific validator error message, and re-invoke the generation model
(Task from generation/) for up to 2 retry attempts before giving up and returning a
graceful failure response to the user. Log every retry (success/failure, error type) so
the evaluation harness can report a "self-correction success rate" metric — this is a
nice additional number for the paper's results section.
```

### Task E — Evaluation Harness

```
Implement evaluation/ as a standalone CLI (evaluate.py) that runs the full benchmark
suite and produces paper-ready tables and plots.

For a given test set (data/{domain}_augmented.jsonl held-out split), for every question:
  1. Run schema linking -> generation -> validation -> rewriting -> execution for BOTH
     Path A (unoptimized/no partition-awareness) and Path B (your partition-aware
     pipeline).
  2. Compute and log: Execution Accuracy (EX), Property Set Jaccard Similarity (PSJS),
     end-to-end latency, cross-partition network hop count, bytes transferred, and
     self-correction retries used.

Aggregate results into:
  - A per-domain summary table (mean +/- std for each metric, Path A vs Path B side by
    side, with a paired statistical significance test — Wilcoxon signed-rank test, since
    latency/hop distributions are unlikely to be normal — report p-values).
  - A latency-vs-query-hop-count scatter plot (matplotlib), Path A vs Path B, to visually
    show where partition-awareness helps most (this is a strong figure for the paper).
  - An ablation table isolating: (a) schema linking with vs without partition metadata,
    (b) generation with vs without partition-annotated targets, (c) full pipeline with
    vs without the cost-estimator re-ranking from Kaggle Notebook 4.
  - Export all tables as both Markdown and LaTeX (booktabs style) so they drop directly
    into the paper.

Also implement a reproducibility manifest: log package versions, random seeds, dataset
hashes, and cluster configuration alongside every run, saved as run_manifest.json — a
reviewer or co-author should be able to exactly reproduce any reported number from this
file alone.
```

### Task F — End-to-End API + Orchestration

```
Implement api/ as a FastAPI service exposing POST /query {nl_question, domain} that runs
the full pipeline (schema linking -> generation -> validation/correction -> rewriting ->
distributed execution -> result formatting) and returns both the final answer and a
metrics object (latency, hops, path used). Wire in Task A's docker-compose so `docker
compose up` brings up the DB cluster + this API together. Add a minimal endpoint
GET /health that verifies connectivity to all shards. Write an integration test that
sends 10 sample questions from the held-out test split through the live API and asserts
non-error responses end-to-end.
```

---

## 5. Execution Order (mapped to your Gantt chart phases)

| Phase (from your Gantt) | What to actually do |
|---|---|
| Phase 1 (Weeks 1–4): Research & Planning | Finalize scope to 2–3 CypherBench domains + JanusGraph/Cassandra backend (fastest to stand up). Run **Phase 0** dataset partition-augmentation now — it's listed as "research" time but it's really your first deliverable. |
| Phase 2 (Weeks 5–8): System Design | Coding agent **Task A** (infra scaffolding + cluster). Finalize the partition-annotation comment format before touching Kaggle, since Notebook 3's targets depend on it. |
| Phase 3 (Weeks 9–12): Model & System Dev | Run **Kaggle Notebooks 1–4** in parallel with coding agent **Tasks B, C, D**. These two tracks are independent until integration — don't block one on the other. |
| Phase 4 (Weeks 13–16): Evaluation | Coding agent **Task E** (evaluation harness) + **Task F** (API) to wire everything together; run the full Path A vs Path B benchmark suite. |
| Phase 5 (Weeks 17–20): Documentation & Submission | Pull tables/plots straight from Task E's LaTeX exports into the manuscript. Target GRADES-NDA @ SIGMOD or GenAIK first (per your own strategic roadmap), then expand for SIGMOD/VLDB/ACL. |

---

## 6. Journal-worthiness checklist (don't skip these — reviewers will ask)

- **Baselines beyond your own Path A**: also report numbers for at least one existing method from your Review-1 "Existing Models" table (e.g. plain Text2Cypher with no partition awareness at all, not just your own non-optimized path) so the comparison isn't self-serving.
- **Statistical significance**: Task E already includes Wilcoxon tests — don't drop this from the paper.
- **Semantic-equivalence proof**: Task B's correctness tests (decomposed vs non-decomposed execution parity) are your defense against "did splitting the query break correctness" — cite this explicitly in the Methods section.
- **Scale honesty**: report at what graph size (node/edge count) the partition-aware advantage starts to appear vs where it's negligible — a single aggregate number invites a reviewer to ask "does this actually matter at small scale," so preempt it.
- **Reproducibility manifest** (Task E) — increasingly expected by SIGMOD/VLDB reproducibility tracks.
- **Ablations, not just the headline number**: Notebook 3's Path A/B ablation, Notebook 4's cost-estimator ablation, and Task D's self-correction rate all give you defensible secondary results, which matters a lot for review at top-tier venues where a single number rarely survives scrutiny.

---

*Everything above assumes JanusGraph+Cassandra as the default backend since it's the more scriptable/dockerizable of your two options; swap to Neo4j Fabric throughout if you'd rather target that (the Kaggle notebooks are backend-agnostic, only Tasks A/C/F need Fabric-specific dispatch code instead of Cassandra/JanusGraph clients).*
