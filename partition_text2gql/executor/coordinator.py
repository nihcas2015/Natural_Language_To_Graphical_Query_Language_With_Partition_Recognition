"""
Distributed Query Coordinator for parallel shard dispatch and sandboxed execution.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Any, Optional, Tuple
from partition_text2gql.executor.shard import GraphShard
from partition_text2gql.executor.aggregator import ResultAggregator
from partition_text2gql.core.types import (
    PhysicalExecutionPlan, ShardSubquery, ExecutionMetrics, PlanType
)
from partition_text2gql.core.errors import ExecutionTimeoutError
from partition_text2gql.core.config import EngineConfig


class QueryCoordinator:
    """
    Orchestrates execution across physical graph shards.
    Dispatches subqueries in parallel, monitors latency, and merges results via ResultAggregator.
    """

    def __init__(self, shards: Dict[int, GraphShard], config: Optional[EngineConfig] = None):
        self.shards = shards
        self.config = config or EngineConfig()

    def execute_plan(self, plan: PhysicalExecutionPlan) -> Tuple[List[Dict[str, Any]], ExecutionMetrics]:
        """Executes a PhysicalExecutionPlan across shards with high-resolution metrics tracking."""
        start_time = time.perf_counter()
        shard_latencies: Dict[int, float] = {}
        subquery_results: Dict[str, List[Dict[str, Any]]] = {}
        bytes_shuffled = 0
        rows_shuffled = 0

        # Execute independent subqueries in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_worker_threads) as executor:
            future_to_sq = {}
            for sq in plan.subqueries:
                shard = self.shards.get(sq.partition_id)
                if not shard:
                    # Fallback to partition 0 if shard not instantiated
                    shard = self.shards.get(0, list(self.shards.values())[0])

                future = executor.submit(self._run_shard_subquery, shard, sq)
                future_to_sq[future] = sq

            for future in as_completed(future_to_sq):
                sq = future_to_sq[future]
                try:
                    res_rows, duration_ms = future.result(timeout=self.config.shard_timeout_seconds)
                    shard_latencies[sq.partition_id] = duration_ms
                    subquery_results[sq.subquery_id] = res_rows

                    # Track shuffle volume (bytes transferred from shard to coordinator)
                    import json
                    row_bytes = len(json.dumps(res_rows).encode('utf-8'))
                    bytes_shuffled += row_bytes
                    rows_shuffled += len(res_rows)

                except FuturesTimeoutError:
                    raise ExecutionTimeoutError(sq.partition_id, self.config.shard_timeout_seconds)

        # Execute Boundary Joins
        merged_rows: List[Dict[str, Any]] = []
        if not plan.boundary_joins and plan.subqueries:
            # Single partition query
            first_sq_id = plan.subqueries[0].subquery_id
            merged_rows = subquery_results.get(first_sq_id, [])
        else:
            for join_spec in plan.boundary_joins:
                left = subquery_results.get(join_spec.left_subquery_id, [])
                right = subquery_results.get(join_spec.right_subquery_id, [])
                merged_rows = ResultAggregator.execute_boundary_join(left, right, join_spec)

        final_rows = ResultAggregator.format_final_results(merged_rows, plan.global_return_clause)
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        metrics = ExecutionMetrics(
            plan_type=plan.plan_type,
            latency_ms=round(total_latency_ms, 2),
            cross_partition_hops=plan.estimated_cross_hops,
            network_shuffle_volume_bytes=bytes_shuffled,
            subqueries_dispatched=len(plan.subqueries),
            shard_latencies_ms=shard_latencies,
            intermediate_rows_shuffled=rows_shuffled,
            final_result_count=len(final_rows)
        )

        return final_rows, metrics

    def _run_shard_subquery(self, shard: GraphShard, sq: ShardSubquery) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        rows = shard.execute_local_query(sq.cypher_text, sq.input_bindings)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return rows, duration_ms

