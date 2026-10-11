"""
Cost-Based Plan Enumerator & Optimizer (Path B).
Orchestrates Statistics Lookup, Cost Estimation, and AST Rewriting to emit the optimal PhysicalExecutionPlan.
"""

from typing import Optional
from partition_text2gql.parser.ast_nodes import QueryAST
from partition_text2gql.core.types import PhysicalExecutionPlan, PlanType
from partition_text2gql.stats.store import PartitionStatisticsStore
from partition_text2gql.optimizer.cost_model import PhysicalCostEstimator
from partition_text2gql.optimizer.rewriter import ASTPartitionRewriter
from partition_text2gql.core.config import EngineConfig


class CostBasedOptimizer:
    """
    Path B Planning Engine:
    Step I: Statistics Lookup (Partition Statistics Store)
    Step II: Cost-Based Estimation & Scoring
    Step III: Plan Enumeration & AST Optimization
    """

    def __init__(self, stats_store: PartitionStatisticsStore, config: Optional[EngineConfig] = None):
        self.stats_store = stats_store
        self.config = config or EngineConfig()
        self.cost_estimator = PhysicalCostEstimator(stats_store, self.config)
        self.rewriter = ASTPartitionRewriter(stats_store)

    def optimize(self, ast: QueryAST, domain: str) -> PhysicalExecutionPlan:
        """Generates the optimal physical distributed execution plan."""
        # 1. Step I: Statistics Lookup happens implicitly through stats_store
        # 2. Step III: AST-Level Partition-Aware Rewriting (decomposition & pushdown)
        plan = self.rewriter.rewrite(ast, domain)

        # 3. Step II: Cost Estimation & Scoring on the rewritten plan
        cost, hops, shuffle = self.cost_estimator.estimate_query_cost(ast, domain)
        plan.estimated_cost = cost
        plan.estimated_cross_hops = hops
        plan.estimated_shuffle_bytes = shuffle

        return plan

