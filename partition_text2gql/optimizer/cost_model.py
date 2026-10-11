"""
Cost-Based Estimation & Scoring for distributed graph queries (Path B - Step II).
Computes formal mathematical physical cost including hop penalties, intermediate network shuffle volume, and local scan latency.
"""

from typing import List, Dict, Optional, Tuple
from partition_text2gql.parser.ast_nodes import QueryAST, MatchClause, NodePattern, RelPattern
from partition_text2gql.stats.store import PartitionStatisticsStore
from partition_text2gql.core.config import EngineConfig


class PhysicalCostEstimator:
    """
    Cost model evaluating physical execution overhead on distributed shards:
    Cost = C_local_compute + C_network_hops + C_network_shuffle
    """

    def __init__(self, stats_store: PartitionStatisticsStore, config: Optional[EngineConfig] = None):
        self.stats_store = stats_store
        self.config = config or EngineConfig()

    def estimate_query_cost(
        self,
        ast: QueryAST,
        domain: str,
        node_partition_bindings: Optional[Dict[str, int]] = None
    ) -> Tuple[float, int, int]:
        """
        Estimates total execution cost, cross-partition hops, and network shuffle bytes for an AST.

        Returns
        -------
        Tuple[float, int, int]
            (total_cost_score, cross_partition_hops, network_shuffle_bytes)
        """
        bindings = node_partition_bindings or {}
        topology = self.stats_store.get_topology(domain)
        avg_partition_size = 10000
        if topology and topology.partition_sizes:
            avg_partition_size = sum(topology.partition_sizes.values()) // max(1, len(topology.partition_sizes))

        cross_hops = 0
        shuffle_bytes = 0
        total_scan_units = 0

        # Estimate per MATCH clause
        for m in ast.match_clauses:
            for path in m.patterns:
                elements = path.elements
                # Traverse path sequence: node - rel - node - rel - node...
                for i in range(len(elements) - 2):
                    e1 = elements[i]
                    e2 = elements[i + 1]
                    e3 = elements[i + 2]

                    if isinstance(e1, NodePattern) and isinstance(e2, RelPattern) and isinstance(e3, NodePattern):
                        # Determine partitions of e1 and e3
                        p1 = e1.partition_id if e1.partition_id is not None else bindings.get(e1.variable)
                        p3 = e3.partition_id if e3.partition_id is not None else bindings.get(e3.variable)

                        # If partitions unresolved, infer from labels
                        if p1 is None and e1.labels:
                            parts1 = self.stats_store.get_partitions_for_label(domain, e1.labels[0])
                            p1 = parts1[0] if parts1 else 0
                        if p3 is None and e3.labels:
                            parts3 = self.stats_store.get_partitions_for_label(domain, e3.labels[0])
                            p3 = parts3[0] if parts3 else 0

                        # Check if boundary crossing occurred
                        if p1 is not None and p3 is not None and p1 != p3:
                            cross_hops += 1
                            # Estimate intermediate result shuffle volume
                            # Base estimate: 100 tuples * 64 bytes/tuple
                            est_intermediate_rows = 100
                            if e1.properties or (m.where and e1.variable and e1.variable in str(m.where)):
                                est_intermediate_rows = 10  # Filter selectivity reduces intermediate size
                            shuffle_bytes += est_intermediate_rows * 64

                        total_scan_units += (avg_partition_size // 100)

        # Total Cost Formula
        cost_score = (
            total_scan_units * self.config.cost_weight_local_scan +
            cross_hops * self.config.cost_weight_hop +
            shuffle_bytes * self.config.cost_weight_byte_shuffle
        )

        return cost_score, cross_hops, shuffle_bytes

