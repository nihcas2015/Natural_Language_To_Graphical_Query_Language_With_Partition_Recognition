"""
AST to openCypher / GQL Serializer.
Reconstructs normalized, human-readable Cypher queries from typed AST nodes.
"""

from typing import Any
from partition_text2gql.parser.ast_nodes import (
    ASTNode, QueryAST, MatchClause, PatternPath, NodePattern, RelPattern,
    RelDirection, ReturnClause, ReturnItem, OrderByItem, WithClause,
    LiteralExpr, VariableExpr, PropertyAccessExpr, BinaryOpExpr,
    UnaryOpExpr, FunctionCallExpr, ListLiteralExpr, MapLiteralExpr, ExprNode
)


class CypherSerializer:
    """Serializes QueryAST and individual AST nodes into Cypher text."""

    @classmethod
    def serialize(cls, node: ASTNode) -> str:
        serializer = cls()
        return serializer.visit(node)

    def visit(self, node: ASTNode) -> str:
        if isinstance(node, QueryAST):
            return self.visit_query(node)
        elif isinstance(node, MatchClause):
            return self.visit_match(node)
        elif isinstance(node, PatternPath):
            return self.visit_pattern_path(node)
        elif isinstance(node, NodePattern):
            return self.visit_node_pattern(node)
        elif isinstance(node, RelPattern):
            return self.visit_rel_pattern(node)
        elif isinstance(node, ReturnClause):
            return self.visit_return_clause(node)
        elif isinstance(node, ReturnItem):
            return self.visit_return_item(node)
        elif isinstance(node, WithClause):
            return self.visit_with_clause(node)
        elif isinstance(node, ExprNode):
            return self.visit_expr(node)
        raise ValueError(f"Unknown AST node type: {type(node).__name__}")

    def visit_query(self, query: QueryAST) -> str:
        clause_strs = []
        for clause in query.clauses:
            clause_strs.append(self.visit(clause))
        return "\n".join(clause_strs)

    def visit_match(self, match: MatchClause) -> str:
        prefix = ""
        if match.partition_annotation is not None:
            prefix = f"// partition:{match.partition_annotation}\n"
        opt = "OPTIONAL " if match.optional else ""
        paths = ", ".join(self.visit_pattern_path(p) for p in match.patterns)
        res = f"{prefix}{opt}MATCH {paths}"
        if match.where:
            res += f" WHERE {self.visit_expr(match.where)}"
        return res

    def visit_pattern_path(self, path: PatternPath) -> str:
        return "".join(self.visit(e) for e in path.elements)

    def visit_node_pattern(self, node: NodePattern) -> str:
        var = node.variable or ""
        labels = "".join(f":{l}" for l in node.labels)
        props = ""
        if node.properties:
            pairs = [f"{k}: {self.visit_expr(v)}" for k, v in node.properties.items()]
            props = " {" + ", ".join(pairs) + "}"
        return f"({var}{labels}{props})"

    def visit_rel_pattern(self, rel: RelPattern) -> str:
        var = rel.variable or ""
        types = f":{':'.join(rel.rel_types)}" if rel.rel_types else ""
        varlength = rel.varlength or ""
        props = ""
        if rel.properties:
            pairs = [f"{k}: {self.visit_expr(v)}" for k, v in rel.properties.items()]
            props = " {" + ", ".join(pairs) + "}"

        body = f"[{var}{types}{varlength}{props}]" if (var or types or varlength or props) else ""

        if rel.direction == RelDirection.OUTGOING:
            return f"-{body}->"
        elif rel.direction == RelDirection.INCOMING:
            return f"<-{body}-"
        else:
            return f"-{body}-"

    def visit_return_clause(self, ret: ReturnClause) -> str:
        dist = "DISTINCT " if ret.distinct else ""
        items = ", ".join(self.visit_return_item(i) for i in ret.items)
        res = f"RETURN {dist}{items}"
        if ret.order_by:
            order_items = []
            for ob in ret.order_by:
                direction = "ASC" if ob.ascending else "DESC"
                order_items.append(f"{self.visit_expr(ob.expression)} {direction}")
            res += f" ORDER BY {', '.join(order_items)}"
        if ret.skip is not None:
            res += f" SKIP {ret.skip}"
        if ret.limit is not None:
            res += f" LIMIT {ret.limit}"
        return res

    def visit_return_item(self, item: ReturnItem) -> str:
        expr = self.visit_expr(item.expression)
        if item.alias:
            return f"{expr} AS {item.alias}"
        return expr

    def visit_with_clause(self, with_clause: WithClause) -> str:
        dist = "DISTINCT " if with_clause.distinct else ""
        items = ", ".join(self.visit_return_item(i) for i in with_clause.items)
        res = f"WITH {dist}{items}"
        if with_clause.where:
            res += f" WHERE {self.visit_expr(with_clause.where)}"
        return res

    def visit_expr(self, expr: ExprNode) -> str:
        if isinstance(expr, LiteralExpr):
            if expr.literal_type == "string":
                return f"'{expr.value}'"
            elif expr.literal_type == "null":
                return "null"
            elif expr.literal_type == "bool":
                return "true" if expr.value else "false"
            return str(expr.value)

        elif isinstance(expr, VariableExpr):
            return expr.name

        elif isinstance(expr, PropertyAccessExpr):
            return f"{expr.variable}.{expr.property_name}"

        elif isinstance(expr, BinaryOpExpr):
            left = self.visit_expr(expr.left)
            right = self.visit_expr(expr.right)
            return f"({left} {expr.op} {right})"

        elif isinstance(expr, UnaryOpExpr):
            operand = self.visit_expr(expr.operand)
            return f"{expr.op} ({operand})"

        elif isinstance(expr, FunctionCallExpr):
            dist = "DISTINCT " if expr.is_distinct else ""
            args = ", ".join(self.visit_expr(a) for a in expr.arguments)
            return f"{expr.function_name}({dist}{args})"

        elif isinstance(expr, ListLiteralExpr):
            items = ", ".join(self.visit_expr(i) for i in expr.items)
            return f"[{items}]"

        elif isinstance(expr, MapLiteralExpr):
            entries = ", ".join(f"{k}: {self.visit_expr(v)}" for k, v in expr.entries.items())
            return f"{{{entries}}}"

        raise ValueError(f"Unknown expression node: {type(expr).__name__}")

