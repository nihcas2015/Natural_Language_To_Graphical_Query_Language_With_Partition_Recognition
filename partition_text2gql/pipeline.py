"""
End-to-End Orchestrator for Partition-Aware Text-to-GQL (Stages 1 through 6).
Provides the master interface for query parsing, optimization, distributed execution, and performance auditing.
"""

from typing import Dict, List, Optional, Any, Tuple
from partition_text2gql.validator.schema_catalog import GraphDomainSchema
from partition_text2gql.validator.schema_catalog import GraphDomainSchema, LabelSchema, RelSchema
from partition_text2gql.validator.validators import (
    SyntaxValidator, SchemaConformanceChecker, AccessSafetyChecker
)
from partition_text2gql.validator.correction import SelfCorrectionFeedbackLoop
from partition_text2gql.stats.store import PartitionStatisticsStore
from partition_text2gql.optimizer.planner import CostBasedOptimizer
from partition_text2gql.executor.shard import GraphShard
from partition_text2gql.executor.dual_engine import DualExecutionEngine
from partition_text2gql.semantic_parser.intent_ner import IntentClassifier, EntityRelationExtractor
from partition_text2gql.semantic_parser.intent_ner import AutoIntentNER, IntentClassifier, EntityRelationExtractor
from partition_text2gql.semantic_parser.schema_linker import PartitionAwareSchemaLinker
from partition_text2gql.generator.backends import AutoText2GQLGenerator
from partition_text2gql.analytics.reporter import AnalyticsReporter
from partition_text2gql.core.types import CostComparisonReport, QueryIntent, EntitySpan, SchemaElement


class PartitionAwareText2GQLEngine:
    """
    Enterprise & Patent-Ready Master Engine.
    Executes the full 6-stage architecture:
      Stage 1: Semantic Parser (Intent, NER, Partition-Aware Schema Linking)
      Stage 2: Partition-Grounded LLM Prompt / Query Synthesis
      Stage 3: Query Validator (Formal AST Syntax, Schema Conformance, Safety)
      Stage 4: Dual-Path Distributed Query Execution (Path A Naive vs Path B Optimized)
      Stage 5: High-Precision Analytics & Network Hop Tracer
      Stage 2: Partition-Grounded openCypher / GQL Generation (LLM / Synthesizer)
      Stage 3: 3-Tier Validator & Self-Correction Feedback Loop (Syntax, Schema, Safety)
      Stage 4: Cost-Based Optimizer (Predicate Pushdown, Boundary Slicing, Cost Ranking)
      Stage 5: Dual-Path Distributed Query Execution (Path A Naive vs Path B Optimized)
      Stage 6: Output & Performance Comparison Audit
    """

    def __init__(
        self,
        domain: str,
        schema: GraphDomainSchema,
        stats_store: PartitionStatisticsStore,
        shards: Dict[int, GraphShard],
        generator_backend: Optional[str] = None
    ):
        self.domain = domain
        self.schema = schema
        self.stats_store = stats_store
        self.shards = shards

        # Stage 1 modules
        # Stage 1: Semantic Parsing
        self.intent_ner = AutoIntentNER()
        self.schema_linker = PartitionAwareSchemaLinker(schema, stats_store)

        # Stage 3 modules
        # Stage 2: Physically-Grounded Generator
        self.generator = AutoText2GQLGenerator(preferred_backend=generator_backend)

        # Stage 3: Validation & Self-Correction Feedback Loop
        self.syntax_validator = SyntaxValidator()
        self.schema_checker = SchemaConformanceChecker(schema)
        self.safety_checker = AccessSafetyChecker(max_hops=6)
        self.correction_loop = SelfCorrectionFeedbackLoop(schema=schema, max_retries=2, safety_hops=6)

        # Stage 4 modules
        # Stage 4: Optimizer
        self.optimizer = CostBasedOptimizer(stats_store)

        # Stage 5: Dual Execution Engine
        self.dual_engine = DualExecutionEngine(shards, self.optimizer)

    def process_natural_language(
        self,
        nl_question: str,
        gold_or_candidate_cypher: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], CostComparisonReport]:
        """
        Processes an end-to-end request from natural language through all 6 stages.

        If gold_or_candidate_cypher is provided, it is used directly; otherwise
        Stage 2 (LLM/generator) synthesizes a query from natural language.
        Returns final result rows and a Path A vs Path B comparison report.
        """
        # -------------------------------------------------------------
        # Stage 1: Semantic Parsing (Intent, NER, Partition-Aware Schema Linking)
        # -------------------------------------------------------------
        intent: QueryIntent = self.intent_ner.classify_intent(nl_question)
        entities: List[EntitySpan] = self.intent_ner.extract_entities(nl_question)
        linked_schema: List[SchemaElement] = self.schema_linker.link(nl_question, top_k=5)

        # -------------------------------------------------------------
        # Stage 2: Partition-Grounded Query Generation (if not supplied)
        # -------------------------------------------------------------
        if gold_or_candidate_cypher:
            candidate_cypher = gold_or_candidate_cypher
        else:
            candidate_cypher, _ = self.generator.generate(
                nl_question=nl_question,
                domain=self.domain,
                schema_elements=linked_schema,
                intent=intent,
                k_partitions=max(1, len(self.shards))
            )

        # -------------------------------------------------------------
        # Stage 3: Query Validator & Self-Correction Feedback Loop
        # -------------------------------------------------------------
        # 3.1 Syntax Validation (formal AST parser)
        ast = self.syntax_validator.validate(candidate_cypher)

        # 3.2 Schema Conformance
        self.schema_checker.validate(ast)

        # 3.3 Security & Safety Check
        self.safety_checker.validate(candidate_cypher, ast)

        # 3.4 Self-Correction Feedback Loop (feeds diagnostics back to generator)
        def generator_repair_fn(prompt: str) -> str:
            repaired_cypher, _ = self.generator.generate(
                nl_question=prompt,
                domain=self.domain,
                schema_elements=linked_schema,
                intent=intent,
                k_partitions=max(1, len(self.shards))
            )
            return repaired_cypher

        correction_result = self.correction_loop.validate_and_repair(
            nl_question=nl_question,
            initial_query=candidate_cypher,
            generator_fn=generator_repair_fn
        )

        if correction_result.is_valid and correction_result.final_ast is not None:
            ast = correction_result.final_ast
            candidate_cypher = correction_result.final_query

        # -------------------------------------------------------------
        # Stage 4 & 5: Dual Execution & Performance Comparison
        # -------------------------------------------------------------
        comparison_report = self.dual_engine.execute_dual_paths(
            raw_cypher=candidate_cypher,
            ast=ast,
            domain=self.domain,
            nl_question=nl_question
        )

        # Retrieve final optimized results via Path B plan
        plan_b = self.optimizer.optimize(ast, self.domain)
        results, _ = self.dual_engine.coordinator.execute_plan(plan_b)

        return results, comparison_report

    @classmethod
    def create_demo(cls, domain: str = "movie") -> "PartitionAwareText2GQLEngine":
        """Factory helper creating a fully populated demo engine with 2 shards."""
        schema = GraphDomainSchema(
            domain=domain,
            labels={
                "Movie": LabelSchema(name="Movie", properties=["title", "release_year"]),
                "Person": LabelSchema(name="Person", properties=["name", "born"])
            },
            relationships={
                "directedBy": RelSchema(type_name="directedBy", source_labels={"Movie"}, target_labels={"Person"})
            }
        )

        stats_store = PartitionStatisticsStore()
        stats_store.register_domain_partitioning(
            domain=domain,
            partition_map={"m_inception": 0, "m_dark_knight": 0, "p_nolan": 1},
            num_partitions=2
        )

        shard_0 = GraphShard(0)
        shard_0.add_node("m_inception", "Movie", {"title": "Inception", "release_year": 2010})
        shard_0.add_node("m_dark_knight", "Movie", {"title": "The Dark Knight", "release_year": 2008})
        shard_0.add_edge("m_inception", "p_nolan", "directedBy")
        shard_0.add_edge("m_dark_knight", "p_nolan", "directedBy")

        shard_1 = GraphShard(1)
        shard_1.add_node("p_nolan", "Person", {"name": "Christopher Nolan", "born": 1970})

        shards = {0: shard_0, 1: shard_1}
        return cls(domain=domain, schema=schema, stats_store=stats_store, shards=shards)
