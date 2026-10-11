"""
Distributed Result Aggregator for boundary joins and final projection.
Merges tabular intermediate results across partition boundaries using hash joins.
"""

from typing import List, Dict, Any, Optional
from partition_text2gql.core.types import BoundaryJoinSpec


class ResultAggregator:
    """
    Executes distributed boundary hash joins and projects final output records.
    Reduces cross-partition merge overhead by operating purely on minimal join projections.
    """

    @classmethod
    def execute_boundary_join(
        cls,
        left_rows: List[Dict[str, Any]],
        right_rows: List[Dict[str, Any]],
        join_spec: BoundaryJoinSpec
    ) -> List[Dict[str, Any]]:
        """Performs distributed in-memory hash join between two shard subquery result sets."""
        if not left_rows or not right_rows:
            return []

        join_key = join_spec.join_keys[0] if join_spec.join_keys else "join_key"

        # Build hash table on smaller side
        if len(left_rows) <= len(right_rows):
            hash_table: Dict[Any, List[Dict[str, Any]]] = {}
            for row in left_rows:
                k = row.get(join_key)
                if k is not None:
                    hash_table.setdefault(k, []).append(row)

            # Probe with right rows
            joined: List[Dict[str, Any]] = []
            for r_row in right_rows:
                k = r_row.get(join_key)
                if k in hash_table:
                    for l_row in hash_table[k]:
                        merged = dict(l_row)
                        merged.update(r_row)
                        joined.append(merged)
            return joined
        else:
            # Build on right, probe with left
            hash_table: Dict[Any, List[Dict[str, Any]]] = {}
            for row in right_rows:
                k = row.get(join_key)
                if k is not None:
                    hash_table.setdefault(k, []).append(row)

            joined = []
            for l_row in left_rows:
                k = l_row.get(join_key)
                if k in hash_table:
                    for r_row in hash_table[k]:
                        merged = dict(l_row)
                        merged.update(r_row)
                        joined.append(merged)
            return joined

    @classmethod
    def format_final_results(
        cls,
        rows: List[Dict[str, Any]],
        return_clause_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Deduplicates and cleans up internal join keys from final user-facing output."""
        cleaned_results = []
        seen = set()

        for r in rows:
            clean_row = {k: v for k, v in r.items() if not k.startswith("anon_") and k != "join_key"}
            # Deduplicate
            row_tuple = tuple(sorted((k, str(v)) for k, v in clean_row.items()))
            if row_tuple not in seen:
                seen.add(row_tuple)
                cleaned_results.append(clean_row)

        return cleaned_results

