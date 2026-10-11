"""
Syntax, Schema Conformance, and Security/Safety Validators for openCypher.
Enforces that queries are syntactically sound, semantically grounded in the graph schema, and strictly read-only.
"""

from typing import Optional, List, Set, Dict
from partition_text2gql.parser.ast_nodes import (
    QueryAST, NodePattern, RelPattern, PropertyAccessExpr, BinaryOpExpr,
    UnaryOpExpr, FunctionCallExpr, ExprNode
)
from partition_text2gql.parser.cypher_parser import CypherParser
from partition_text2gql.validator.schema_catalog import GraphDomainSchema
from partition_text2gql.core.errors import (
    CypherSyntaxError, SchemaValidationError, SecurityViolationError
)


class SyntaxValidator:
    """Stage 3.1: Validates formal openCypher syntax and returns parsed QueryAST."""

    @classmethod
    def validate(cls, cypher_text: str) -> QueryAST:
        """Parses and validates syntax; raises CypherSyntaxError on failure."""
        return CypherParser.parse_query(cypher_text)


class SchemaConformanceChecker:
    """Stage 3.2: Verifies that referenced labels, relationship types, and properties exist in domain schema."""

    def __init__(self, schema: GraphDomainSchema):
        self.schema = schema

    def validate(self, ast: QueryAST) -> None:
        """Walks AST and verifies every label, relationship type, and property against schema catalog."""
        # 1. Map variables to their node labels
        var_to_labels: Dict[str, List[str]] = {}
        for node in ast.get_all_node_patterns():
            if node.variable and node.labels:
                var_to_labels[node.variable] = node.labels

            # Check node labels
            for label in node.labels:
                if not self.schema.has_label(label):
                    raise SchemaValidationError(
                        message=f"Node label '{label}' does not exist in domain schema.",
                        element_type="NodeLabel",
                        element_name=label,
                        domain=self.schema.domain
                    )

            # Check node inline properties
            for prop_name in node.properties.keys():
                found = any(self.schema.has_property(l, prop_name) for l in node.labels) if node.labels else self.schema.has_property(None, prop_name)
                if not found:
                    raise SchemaValidationError(
                        message=f"Property '{prop_name}' does not exist for labels {node.labels}.",
                        element_type="Property",
                        element_name=prop_name,
                        domain=self.schema.domain
                    )

        # 2. Check relationship types
        for rel in ast.get_all_rel_patterns():
            for rtype in rel.rel_types:
                if not self.schema.has_relationship(rtype):
                    raise SchemaValidationError(
                        message=f"Relationship type '{rtype}' does not exist in domain schema.",
                        element_type="RelationshipType",
                        element_name=rtype,
                        domain=self.schema.domain
                    )

        # 3. Check properties in WHERE and RETURN expressions
        for m in ast.match_clauses:
            if m.where:
                self._check_expression_properties(m.where, var_to_labels)

        if ast.return_clause:
            for item in ast.return_clause.items:
                self._check_expression_properties(item.expression, var_to_labels)

    def _check_expression_properties(self, expr: ExprNode, var_to_labels: Dict[str, List[str]]) -> None:
        if isinstance(expr, PropertyAccessExpr):
            labels = var_to_labels.get(expr.variable, [])
            valid = any(self.schema.has_property(l, expr.property_name) for l in labels) if labels else self.schema.has_property(None, expr.property_name)
            if not valid:
                raise SchemaValidationError(
                    message=f"Property '{expr.property_name}' does not exist in domain schema.",
                    element_type="Property",
                    element_name=expr.property_name,
                    domain=self.schema.domain
                )
        elif isinstance(expr, BinaryOpExpr):
            self._check_expression_properties(expr.left, var_to_labels)
            self._check_expression_properties(expr.right, var_to_labels)
        elif isinstance(expr, UnaryOpExpr):
            self._check_expression_properties(expr.operand, var_to_labels)
        elif isinstance(expr, FunctionCallExpr):
            for arg in expr.arguments:
                self._check_expression_properties(arg, var_to_labels)


class AccessSafetyChecker:
    """Stage 3.3: Enforces security policies (read-only execution, hop limits, memory bomb prevention)."""

    FORBIDDEN_KEYWORDS = {
        "CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DROP", "ALTER", "DETACH", "INDEX", "CONSTRAINT"
    }

    def __init__(self, max_hops: int = 6):
        self.max_hops = max_hops

    def validate_text(self, cypher_text: str) -> None:
        """Enforces read-only safety on raw query string before parsing."""
        upper_text = cypher_text.upper()
        for kw in self.FORBIDDEN_KEYWORDS:
            tokens = upper_text.split()
            if kw in tokens or f"{kw} " in upper_text:
                raise SecurityViolationError(
                    f"Forbidden mutating keyword '{kw}' detected. System is read-only.",
                    violation_code="FORBIDDEN_MUTATION_OP"
                )

    def validate_ast(self, ast: QueryAST) -> None:
        """Enforces hop limits and detects unbounded wildcard expansions in the AST."""
        total_hops = 0
        for m in ast.match_clauses:
            for pattern in m.patterns:
                rels = pattern.relationships
                total_hops += len(rels)

                for rel in rels:
                    if rel.varlength:
                        vl = rel.varlength.strip("*")
                        if vl == "" or vl.endswith("..") or vl == "*":
                            raise SecurityViolationError(
                                f"Unbounded variable-length traversal '{rel.varlength}' rejected (DoS risk). Must specify upper bound.",
                                violation_code="UNBOUNDED_TRAVERSAL"
                            )
                        if ".." in vl:
                            upper_b = vl.split("..")[1].strip()
                            if upper_b and int(upper_b) > self.max_hops:
                                raise SecurityViolationError(
                                    f"Variable-length expansion {upper_b} exceeds maximum allowed hop limit ({self.max_hops}).",
                                    violation_code="MAX_HOPS_EXCEEDED"
                                )

        if total_hops > self.max_hops:
            raise SecurityViolationError(
                f"Query contains {total_hops} hops, exceeding maximum allowed hop limit ({self.max_hops}).",
                violation_code="MAX_HOPS_EXCEEDED"
            )

    def validate(self, cypher_text: str, ast: QueryAST) -> None:
        self.validate_text(cypher_text)
        self.validate_ast(ast)
