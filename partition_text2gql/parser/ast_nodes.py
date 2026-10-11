"""
Formally typed Abstract Syntax Tree (AST) definitions for openCypher / GQL.
Enables syntax trees to be inspected, partitioned, rewritten, and serialized without text-hack fragility.
"""

from abc import ABC
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Union


class RelDirection(Enum):
    OUTGOING = auto()    # (a)-[r]->(b)
    INCOMING = auto()    # (a)<-[r]-(b)
    UNDIRECTED = auto()  # (a)-[r]-(b)


class ASTNode(ABC):
    """Abstract base class for all Cypher AST nodes."""
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit(self)


# ==========================================
# Expressions
# ==========================================

class ExprNode(ASTNode):
    """Base class for all expression nodes."""
    pass


class LiteralExpr(ExprNode):
    def __init__(self, value: Any, literal_type: str):
        self.value = value
        self.literal_type = literal_type  # 'string', 'int', 'float', 'bool', 'null'

    def __repr__(self) -> str:
        return f"Literal({repr(self.value)})"


class VariableExpr(ExprNode):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"Var({self.name})"


class PropertyAccessExpr(ExprNode):
    def __init__(self, variable: str, property_name: str):
        self.variable = variable
        self.property_name = property_name

    def __repr__(self) -> str:
        return f"Prop({self.variable}.{self.property_name})"


class BinaryOpExpr(ExprNode):
    def __init__(self, left: ExprNode, op: str, right: ExprNode):
        self.left = left
        self.op = op  # '=', '<>', '!=', '<', '<=', '>', '>=', 'AND', 'OR', '+', '-', '*', '/'
        self.right = right

    def __repr__(self) -> str:
        return f"BinaryOp({self.left} {self.op} {self.right})"


class UnaryOpExpr(ExprNode):
    def __init__(self, op: str, operand: ExprNode):
        self.op = op  # 'NOT', '-'
        self.operand = operand

    def __repr__(self) -> str:
        return f"UnaryOp({self.op} {self.operand})"


class FunctionCallExpr(ExprNode):
    def __init__(self, function_name: str, arguments: List[ExprNode], is_distinct: bool = False):
        self.function_name = function_name.upper()
        self.arguments = arguments
        self.is_distinct = is_distinct

    def __repr__(self) -> str:
        dist = "DISTINCT " if self.is_distinct else ""
        return f"Func({self.function_name}({dist}{', '.join(repr(a) for a in self.arguments)}))"


class ListLiteralExpr(ExprNode):
    def __init__(self, items: List[ExprNode]):
        self.items = items

    def __repr__(self) -> str:
        return f"List({self.items})"


class MapLiteralExpr(ExprNode):
    def __init__(self, entries: Dict[str, ExprNode]):
        self.entries = entries

    def __repr__(self) -> str:
        return f"Map({self.entries})"


# ==========================================
# Graph Pattern Elements
# ==========================================

class NodePattern(ASTNode):
    """(variable:Label1:Label2 {prop: val})"""
    def __init__(
        self,
        variable: Optional[str] = None,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, ExprNode]] = None,
        partition_id: Optional[int] = None,
    ):
        self.variable = variable
        self.labels = labels or []
        self.properties = properties or {}
        self.partition_id = partition_id  # Assigned by schema-linker or AST rewriter

    def __repr__(self) -> str:
        var = self.variable or ""
        labels_str = "".join(f":{l}" for l in self.labels)
        part_str = f" [part={self.partition_id}]" if self.partition_id is not None else ""
        props = f" {self.properties}" if self.properties else ""
        return f"Node({var}{labels_str}{props}{part_str})"


class RelPattern(ASTNode):
    """-[variable:TYPE {prop: val}]-> or <-[...]-, etc."""
    def __init__(
        self,
        variable: Optional[str] = None,
        rel_types: Optional[List[str]] = None,
        properties: Optional[Dict[str, ExprNode]] = None,
        direction: RelDirection = RelDirection.OUTGOING,
        varlength: Optional[str] = None,  # e.g. '*1..3'
    ):
        self.variable = variable
        self.rel_types = rel_types or []
        self.properties = properties or {}
        self.direction = direction
        self.varlength = varlength

    def __repr__(self) -> str:
        arrow = "->" if self.direction == RelDirection.OUTGOING else ("<-" if self.direction == RelDirection.INCOMING else "-")
        types_str = "|".join(self.rel_types) if self.rel_types else ""
        return f"Rel({arrow} [:{types_str}])"


class PatternPath(ASTNode):
    """Sequence: NodePattern, RelPattern, NodePattern, RelPattern, NodePattern..."""
    def __init__(self, elements: List[Union[NodePattern, RelPattern]]):
        self.elements = elements

    @property
    def nodes(self) -> List[NodePattern]:
        return [e for e in self.elements if isinstance(e, NodePattern)]

    @property
    def relationships(self) -> List[RelPattern]:
        return [e for e in self.elements if isinstance(e, RelPattern)]

    def __repr__(self) -> str:
        return f"PatternPath({' '.join(repr(e) for e in self.elements)})"


# ==========================================
# Clauses
# ==========================================

class ClauseNode(ASTNode):
    """Base class for query clauses."""
    pass


class MatchClause(ClauseNode):
    """MATCH (a)-[r]->(b) WHERE a.val = 1"""
    def __init__(
        self,
        patterns: List[PatternPath],
        where: Optional[ExprNode] = None,
        optional: bool = False,
        partition_annotation: Optional[int] = None,
    ):
        self.patterns = patterns
        self.where = where
        self.optional = optional
        self.partition_annotation = partition_annotation

    def __repr__(self) -> str:
        opt = "OPTIONAL " if self.optional else ""
        part = f" // partition:{self.partition_annotation}\n" if self.partition_annotation is not None else ""
        where_str = f" WHERE {self.where}" if self.where else ""
        return f"{part}{opt}MATCH {self.patterns}{where_str}"


class WhereClause(ClauseNode):
    """Standalone WHERE clause or attached to WITH."""
    def __init__(self, expression: ExprNode):
        self.expression = expression

    def __repr__(self) -> str:
        return f"WHERE {self.expression}"


class ReturnItem(ASTNode):
    def __init__(self, expression: ExprNode, alias: Optional[str] = None):
        self.expression = expression
        self.alias = alias

    def __repr__(self) -> str:
        alias_str = f" AS {self.alias}" if self.alias else ""
        return f"{self.expression}{alias_str}"


class OrderByItem(ASTNode):
    def __init__(self, expression: ExprNode, ascending: bool = True):
        self.expression = expression
        self.ascending = ascending

    def __repr__(self) -> str:
        return f"{self.expression} {'ASC' if self.ascending else 'DESC'}"


class ReturnClause(ClauseNode):
    """RETURN DISTINCT a.name, COUNT(b) ORDER BY a.name DESC SKIP 10 LIMIT 5"""
    def __init__(
        self,
        items: List[ReturnItem],
        distinct: bool = False,
        order_by: Optional[List[OrderByItem]] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        self.items = items
        self.distinct = distinct
        self.order_by = order_by or []
        self.skip = skip
        self.limit = limit

    def __repr__(self) -> str:
        dist = "DISTINCT " if self.distinct else ""
        items_str = ", ".join(repr(i) for i in self.items)
        return f"RETURN {dist}{items_str}"


class WithClause(ClauseNode):
    """WITH a, COUNT(b) AS cnt WHERE cnt > 5"""
    def __init__(
        self,
        items: List[ReturnItem],
        distinct: bool = False,
        where: Optional[ExprNode] = None,
    ):
        self.items = items
        self.distinct = distinct
        self.where = where

    def __repr__(self) -> str:
        return f"WITH {self.items}"


class QueryAST(ASTNode):
    """Complete rooted AST representing an entire openCypher query."""
    def __init__(self, clauses: List[ClauseNode]):
        self.clauses = clauses

    @property
    def match_clauses(self) -> List[MatchClause]:
        return [c for c in self.clauses if isinstance(c, MatchClause)]

    @property
    def return_clause(self) -> Optional[ReturnClause]:
        for c in self.clauses:
            if isinstance(c, ReturnClause):
                return c
        return None

    def get_all_node_patterns(self) -> List[NodePattern]:
        nodes = []
        for m in self.match_clauses:
            for p in m.patterns:
                nodes.extend(p.nodes)
        return nodes

    def get_all_rel_patterns(self) -> List[RelPattern]:
        rels = []
        for m in self.match_clauses:
            for p in m.patterns:
                rels.extend(p.relationships)
        return rels

    def get_referenced_labels(self) -> List[str]:
        labels = set()
        for node in self.get_all_node_patterns():
            labels.update(node.labels)
        return sorted(labels)

    def get_referenced_rel_types(self) -> List[str]:
        rel_types = set()
        for rel in self.get_all_rel_patterns():
            rel_types.update(rel.rel_types)
        return sorted(rel_types)

    def __repr__(self) -> str:
        return f"QueryAST({self.clauses})"


class ASTVisitor:
    """Base visitor pattern for traversing and transforming Cypher ASTs."""
    def visit(self, node: ASTNode) -> Any:
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode) -> Any:
        raise NotImplementedError(f"No visitor method visit_{type(node).__name__}")

