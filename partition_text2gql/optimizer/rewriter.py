"""
AST-Level Partition-Aware Query Rewriting Engine (Path B - Step III).
Decomposes cross-partition MATCH trees into localized subqueries, executes predicate pushdown, and synthesizes minimal boundary-join operators.
"""

from typing import List, Dict, Optional, Tuple, Set, Any
from partition_text2gql.parser.ast_nodes import (
    QueryAST, MatchClause, PatternPath, NodePattern, RelPattern, ReturnClause,
    ReturnItem, BinaryOpExpr, PropertyAccessExpr, VariableExpr, ExprNode, RelDirection
)
from partition_text2gql.parser.serializer import CypherSerializer
from partition_text2gql.core.types import (
    PhysicalExecutionPlan, PlanType, ShardSubquery, BoundaryJoinSpec
)
from partition_text2gql.stats.store import PartitionStatisticsStore


class ASTPartitionRewriter:
    """
    Core Optimizer:
    Transforms a global multi-hop openCypher query AST into a decomposed, parallel, partition-localized PhysicalExecutionPlan.
    """

    def __init__(self, stats_store: PartitionStatisticsStore):
        self.stats_store = stats_store

    def rewrite(self, ast: QueryAST, domain: str) -> PhysicalExecutionPlan:
        """Transforms AST into an optimized distributed execution plan."""
        # 1. Resolve partition assignment for each node pattern
        node_partitions: Dict[str, int] = self._resolve_node_partitions(ast, domain)

        # 2. Check if all nodes fall into a single partition
        all_pids = set(node_partitions.values())
        if len(all_pids) <= 1:
            # Single-partition query: dispatch directly without boundary joins
            single_pid = list(all_pids)[0] if all_pids else 0
            subquery = ShardSubquery(
                subquery_id="sq_0",
                partition_id=single_pid,
                cypher_text=CypherSerializer.serialize(ast),
                projected_join_keys=[],
                projected_payload_keys=[item.alias or str(item.expression) for item in (ast.return_clause.items if ast.return_clause else [])]
            )
            return PhysicalExecutionPlan(
                plan_id=f"plan_{domain}_single",
                plan_type=PlanType.PATH_B_OPTIMIZED,
                subqueries=[subquery],
                boundary_joins=[],
                global_return_clause=CypherSerializer.serialize(ast.return_clause) if ast.return_clause else "RETURN *",
                estimated_cost=1.0,
                estimated_shuffle_bytes=0,
                estimated_cross_hops=0
            )

        # 3. Cross-partition query: decompose across cut boundary
        return self._decompose_cross_partition_query(ast, domain, node_partitions)

    def _resolve_node_partitions(self, ast: QueryAST, domain: str) -> Dict[str, int]:
        """Resolves physical partition for each node variable in the AST."""
        assignments: Dict[str, int] = {}
        top = self.stats_store.get_topology(domain)

        for m in ast.match_clauses:
            clause_part = m.partition_annotation

            for path in m.patterns:
                for node in path.nodes:
                    var = node.variable or f"anon_{id(node)}"

                    # If clause explicitly annotated with partition
                    if clause_part is not None:
                        assignments[var] = clause_part
                        continue

                    # If node has partition_id
                    if node.partition_id is not None:
                        assignments[var] = node.partition_id
                        continue

                    # Check properties for entity ID (e.g. name: 'X')
                    assigned = False
                    if node.properties:
                        for prop_val in node.properties.values():
                            val_str = str(getattr(prop_val, 'value', ''))
                            pid = self.stats_store.get_node_partition(domain, val_str)
                            if pid is not None:
                                assignments[var] = pid
                                assigned = True
                                break

                    if not assigned:
                        # Fallback by label distribution
                        if node.labels:
                            lbl = node.labels[0]
                            pids = self.stats_store.get_partitions_for_label(domain, lbl)
                            assignments[var] = pids[0] if pids else 0
                        else:
                            assignments[var] = 0

        return assignments

    def _decompose_cross_partition_query(
        self,
        ast: QueryAST,
        domain: str,
        node_partitions: Dict[str, int]
    ) -> PhysicalExecutionPlan:
        """Splits multi-hop pattern across shard boundaries with predicate pushdown."""
        match_clause = ast.match_clauses[0]
        path = match_clause.patterns[0]
        elements = path.elements

        # Identify boundary cut edge
        cut_idx = -1
        p_left = -1
        p_right = -1

        for i in range(len(elements) - 2):
            e1 = elements[i]
            e2 = elements[i + 1]
            e3 = elements[i + 2]
            if isinstance(e1, NodePattern) and isinstance(e2, RelPattern) and isinstance(e3, NodePattern):
                pid1 = node_partitions.get(e1.variable, 0)
                pid3 = node_partitions.get(e3.variable, 1)
                if pid1 != pid3:
                    cut_idx = i + 1  # RelPattern index
                    p_left = pid1
                    p_right = pid3
                    break

        if cut_idx == -1:
            # Fallback to single partition if no explicit cut found
            cut_idx = 1
            p_left = 0
            p_right = 1

        # Decompose into Left and Right Sub-Paths
        left_elements = elements[:cut_idx]
        cut_rel: RelPattern = elements[cut_idx]
        right_elements = elements[cut_idx + 1:]

        # Identify Join Variable (the boundary vertex)
        left_boundary_node: NodePattern = left_elements[-1]
        right_boundary_node: NodePattern = right_elements[0]
        join_var = right_boundary_node.variable or "join_target"

        # Predicate Pushdown: push WHERE conditions to their owning partition
        left_where, right_where = self._pushdown_predicates(
            match_clause.where,
            set(e.variable for e in left_elements if isinstance(e, NodePattern) and e.variable),
            set(e.variable for e in right_elements if isinstance(e, NodePattern) and e.variable)
        )

        # Construct Subquery 1 (Left Shard): traverses up through the cut edge to target ID
        left_path_elements = list(left_elements) + [cut_rel, NodePattern(variable=join_var)]
        left_path = PatternPath(elements=left_path_elements)
        left_subquery_cypher = f"MATCH {CypherSerializer.serialize(left_path)}"
        if left_where:
            left_subquery_cypher += f" WHERE {CypherSerializer.serialize(left_where)}"
        left_subquery_cypher += f" RETURN DISTINCT {join_var} AS join_key"

        subquery_left = ShardSubquery(
            subquery_id="sq_left",
            partition_id=p_left,
            cypher_text=left_subquery_cypher,
            projected_join_keys=["join_key"],
            projected_payload_keys=["join_key"]
        )

        # Construct Subquery 2 (Right Shard)
        right_path = PatternPath(elements=right_elements)
        right_subquery_cypher = f"MATCH {CypherSerializer.serialize(right_path)}"
        if right_where:
            right_subquery_cypher += f" WHERE {CypherSerializer.serialize(right_where)}"
        
        # Include original return items or project right variables
        right_ret_items = []
        if ast.return_clause:
            for item in ast.return_clause.items:
                right_ret_items.append(CypherSerializer.serialize(item))
        else:
            right_ret_items.append(f"{join_var_right} AS join_key")

        if f"{join_var} AS join_key" not in right_ret_items and "join_key" not in right_ret_items:
            right_ret_items.insert(0, f"{join_var} AS join_key")

        right_subquery_cypher += f" RETURN {', '.join(right_ret_items)}"

        subquery_right = ShardSubquery(
            subquery_id="sq_right",
            partition_id=p_right,
            cypher_text=right_subquery_cypher,
            projected_join_keys=["join_key"],
            projected_payload_keys=right_ret_items,
            depends_on=["sq_left"]
        )

        # Boundary Join Specification
        boundary_join = BoundaryJoinSpec(
            left_subquery_id="sq_left",
            right_subquery_id="sq_right",
            join_keys=["join_key"],
            join_type="INNER"
        )

        global_ret = CypherSerializer.serialize(ast.return_clause) if ast.return_clause else "RETURN *"

        return PhysicalExecutionPlan(
            plan_id=f"plan_{domain}_decomposed",
            plan_type=PlanType.PATH_B_OPTIMIZED,
            subqueries=[subquery_left, subquery_right],
            boundary_joins=[boundary_join],
            global_return_clause=global_ret,
            estimated_cost=25.0,
            estimated_shuffle_bytes=256,
            estimated_cross_hops=1
        )

    def _pushdown_predicates(
        self,
        where_expr: Optional[ExprNode],
        left_vars: Set[str],
        right_vars: Set[str]
    ) -> Tuple[Optional[ExprNode], Optional[ExprNode]]:
        """Separates compound WHERE expressions into shard-local pushdown filters."""
        if not where_expr:
            return None, None

        if isinstance(where_expr, BinaryOpExpr) and where_expr.op == "AND":
            ll, lr = self._pushdown_predicates(where_expr.left, left_vars, right_vars)
            rl, rr = self._pushdown_predicates(where_expr.right, left_vars, right_vars)
            
            left_combined = None
            if ll and rl:
                left_combined = BinaryOpExpr(ll, "AND", rl)
            else:
                left_combined = ll or rl

            right_combined = None
            if lr and rr:
                right_combined = BinaryOpExpr(lr, "AND", rr)
            else:
                right_combined = lr or rr

            return left_combined, right_combined

        # Leaf predicate
        vars_in_expr = self._extract_variables(where_expr)
        if vars_in_expr.issubset(left_vars):
            return where_expr, None
        elif vars_in_expr.issubset(right_vars):
            return None, where_expr

        # Cross-boundary predicate (must be evaluated in coordinator)
        return None, None

    def _extract_variables(self, expr: ExprNode) -> Set[str]:
        vars_found = set()
        if isinstance(expr, VariableExpr):
            vars_found.add(expr.name)
        elif isinstance(expr, PropertyAccessExpr):
            vars_found.add(expr.variable)
        elif isinstance(expr, BinaryOpExpr):
            vars_found.update(self._extract_variables(expr.left))
            vars_found.update(self._extract_variables(expr.right))
        return vars_found
