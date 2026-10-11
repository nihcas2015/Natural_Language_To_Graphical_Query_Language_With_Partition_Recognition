"""
Physical Shard / Distributed Graph Engine Substrate.
Represents an isolated physical partition hosting a localized subgraph and local query index.
Executes openCypher subqueries directly against the graph substrate using the formal AST.
"""

import time
import networkx as nx
from typing import Dict, List, Any, Optional, Set
import time
from typing import Dict, List, Any, Optional, Set, Tuple

from partition_text2gql.parser import CypherParser
from partition_text2gql.parser.ast_nodes import (
    QueryAST, MatchClause, PatternPath, NodePattern, RelPattern, RelDirection,
    ExprNode, LiteralExpr, VariableExpr, PropertyAccessExpr, BinaryOpExpr,
    UnaryOpExpr, FunctionCallExpr, ListLiteralExpr
)


def eval_ast_expr(expr: Optional[ExprNode], env: Dict[str, Any], graph: nx.DiGraph) -> Any:
    """
    Evaluates an AST expression in the context of node bindings and properties.
    env maps variable_name -> node_id (or literal value)
    """
    if expr is None:
        return True

    if isinstance(expr, LiteralExpr):
        return expr.value

    elif isinstance(expr, VariableExpr):
        return env.get(expr.name)

    elif isinstance(expr, PropertyAccessExpr):
        var_name = expr.variable
        prop_name = expr.property_name
        node_id = env.get(var_name)
        if node_id is not None and node_id in graph.nodes:
            return graph.nodes[node_id].get(prop_name)
        # Check if direct dictionary in env
        if isinstance(node_id, dict):
            return node_id.get(prop_name)
        return env.get(f"{var_name}.{prop_name}")

    elif isinstance(expr, BinaryOpExpr):
        op = expr.op.upper()
        left_val = eval_ast_expr(expr.left, env, graph)
        right_val = eval_ast_expr(expr.right, env, graph)

        if op in ("=", "=="):
            if left_val is None or right_val is None:
                return left_val == right_val
            # Type-tolerant string/number comparison
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                return left_val == right_val
            return str(left_val).strip().lower() == str(right_val).strip().lower()

        elif op in ("<>", "!="):
            if left_val is None or right_val is None:
                return left_val != right_val
            return str(left_val).strip().lower() != str(right_val).strip().lower()

        elif op == "<":
            return left_val is not None and right_val is not None and left_val < right_val
        elif op == "<=":
            return left_val is not None and right_val is not None and left_val <= right_val
        elif op == ">":
            return left_val is not None and right_val is not None and left_val > right_val
        elif op == ">=":
            return left_val is not None and right_val is not None and left_val >= right_val

        elif op == "AND":
            return bool(left_val) and bool(right_val)
        elif op == "OR":
            return bool(left_val) or bool(right_val)
        elif op == "IN":
            if right_val is None:
                return False
            if isinstance(right_val, (list, tuple, set)):
                return left_val in right_val
            return str(left_val).lower() in str(right_val).lower()
        elif op == "CONTAINS":
            if left_val is None or right_val is None:
                return False
            return str(right_val).lower() in str(left_val).lower()
        elif op == "+":
            return left_val + right_val
        elif op == "-":
            return left_val - right_val
        elif op == "*":
            return left_val * right_val
        elif op == "/":
            return left_val / right_val if right_val != 0 else 0

        return False

    elif isinstance(expr, UnaryOpExpr):
        op = expr.op.upper()
        val = eval_ast_expr(expr.operand, env, graph)
        if op in ("NOT", "!"):
            return not bool(val)
        elif op == "-":
            return -val if val is not None else None
        return val

    elif isinstance(expr, ListLiteralExpr):
        return [eval_ast_expr(it, env, graph) for it in expr.items]

    elif isinstance(expr, FunctionCallExpr):
        fn = expr.function_name.upper()
        args = [eval_ast_expr(a, env, graph) for a in expr.arguments]
        if fn == "TOLOWER" and args:
            return str(args[0]).lower() if args[0] is not None else ""
        elif fn == "TOUPPER" and args:
            return str(args[0]).upper() if args[0] is not None else ""
        elif fn in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
            return args[0] if args else None

    return True


class GraphShard:
    """
    Physical Shard hosting Partition k of the distributed property graph.
    Simulates physical storage node (e.g. Cassandra shard or Neo4j Fabric shard database).
    """

    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.graph = nx.DiGraph()
        self._label_index: Dict[str, Set[str]] = {}

    def add_node(self, node_id: str, label: str, properties: Optional[Dict[str, Any]] = None) -> None:
        props = properties or {}
        self.graph.add_node(node_id, label=label, **props)
        if label not in self._label_index:
            self._label_index[label] = set()
        self._label_index[label].add(node_id)

    def add_edge(self, source_id: str, target_id: str, rel_type: str, properties: Optional[Dict[str, Any]] = None) -> None:
        props = properties or {}
        self.graph.add_edge(source_id, target_id, rel_type=rel_type, **props)

    def _match_node_pattern(self, node_pat: NodePattern) -> Set[str]:
        """Finds candidate node IDs matching label and inline properties."""
        if node_pat.labels:
            candidates: Optional[Set[str]] = None
            for lbl in node_pat.labels:
                lbl_nodes = self._label_index.get(lbl, set())
                candidates = lbl_nodes if candidates is None else (candidates & lbl_nodes)
            cands = candidates if candidates is not None else set()
        else:
            cands = set(self.graph.nodes())

        if not node_pat.properties:
            return cands

        # Filter by inline properties
        filtered = set()
        for nid in cands:
            ndata = self.graph.nodes[nid]
            matched = True
            for pk, p_expr in node_pat.properties.items():
                target_val = eval_ast_expr(p_expr, {node_pat.variable or "n": nid}, self.graph)
                actual_val = ndata.get(pk)
                if str(actual_val).strip().lower() != str(target_val).strip().lower():
                    matched = False
                    break
            if matched:
                filtered.add(nid)
        return filtered

    def execute_local_query(
        self,
        cypher_text: str,
        input_bindings: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a localized openCypher subquery on this shard's subgraph.
        Handles single-node MATCH, two-node MATCH (path traversal), and WHERE filtering.
        Returns tabular rows as List[Dict[str, Any]].
        """
        bindings = input_bindings or {}
        results: List[Dict[str, Any]] = []

        from partition_text2gql.parser import CypherParser

        try:
            ast = CypherParser.parse_query(cypher_text)
            if not ast.match_clauses:
                return results
            match_clause = ast.match_clauses[0]
            if not match_clause.patterns:
                return results
            pattern = match_clause.patterns[0]
            nodes = pattern.nodes
            rels = pattern.relationships

            # Case 1: Single Node Pattern (e.g. MATCH (m:Movie) WHERE m.title = "Inception")
            if len(nodes) == 1 and len(rels) == 0:
                node_pat = nodes[0]
                var_name = node_pat.variable or "n"
                candidates = self._match_node_pattern(node_pat)

                # Apply WHERE filter using formal AST expression evaluator
                if match_clause.where:
                    filtered = set()
                    for nid in candidates:
                        ndata = self.graph.nodes[nid]
                        env = {var_name: nid}
                        for k, v in ndata.items():
                            env[f"{var_name}.{k}"] = v
                        try:
                            if eval_ast_expr(match_clause.where, env, self.graph):
                                filtered.add(nid)
                        except Exception:
                            filtered.add(nid)
                    candidates = filtered

                for nid in candidates:
                    ndata = self.graph.nodes[nid]
                    row = {"join_key": nid, var_name: nid}
                    for k, v in ndata.items():
                        row[f"{var_name}.{k}"] = v
                    results.append(row)
                return results

            # Case 2: Multi-node Pattern (e.g. MATCH (m:Movie)-[r:directedBy]->(p:Person))
            if len(nodes) >= 2:
                n1 = nodes[0]
                n2 = nodes[1]
                r = rels[0] if rels else None
                var1 = n1.variable or "n1"
                var2 = n2.variable or "n2"
                cands1 = self._match_node_pattern(n1)

                if var1 in bindings:
                    bval = bindings[var1]
                    bset = set(bval) if isinstance(bval, (list, set, tuple)) else {bval}
                    cands1 = cands1 & bset

                for u in cands1:
                    u_data = self.graph.nodes[u]
                    neighbors = []
                    if r is None or r.direction in (RelDirection.OUTGOING, RelDirection.UNDIRECTED):
                        for v in self.graph.successors(u):
                            edata = self.graph.get_edge_data(u, v, default={})
                            rtype = edata.get("rel_type") or edata.get("label")
                            if r is None or not r.rel_types or (rtype and rtype in r.rel_types):
                                neighbors.append(v)
                    if r and r.direction in (RelDirection.INCOMING, RelDirection.UNDIRECTED):
                        for v in self.graph.predecessors(u):
                            edata = self.graph.get_edge_data(v, u, default={})
                            rtype = edata.get("rel_type") or edata.get("label")
                            if r is None or not r.rel_types or (rtype and rtype in r.rel_types):
                                neighbors.append(v)

                    cands2 = self._match_node_pattern(n2)
                    if var2 in bindings:
                        bval = bindings[var2]
                        bset = set(bval) if isinstance(bval, (list, set, tuple)) else {bval}
                        cands2 = cands2 & bset

                    valid_targets = [v for v in neighbors if v in cands2]
                    for v in valid_targets:
                        v_data = self.graph.nodes.get(v, {})
                        env = {var1: u, var2: v}
                        for k, val in u_data.items():
                            env[f"{var1}.{k}"] = val
                        for k, val in v_data.items():
                            env[f"{var2}.{k}"] = val

                        if match_clause.where and not eval_ast_expr(match_clause.where, env, self.graph):
                            continue

                        row = {"join_key": v, var1: u, var2: v}
                        for k, val in u_data.items():
                            row[f"{var1}.{k}"] = val
                        for k, val in v_data.items():
                            row[f"{var2}.{k}"] = val
                        results.append(row)

                return results

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Shard {self.partition_id} query failed: {e}")
            raise

        return results

