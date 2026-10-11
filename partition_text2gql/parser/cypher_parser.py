"""
Formal Recursive-Descent openCypher / GQL Parser.
Zero regex: constructs typed AST nodes directly with rigorous syntax verification and operator precedence.
"""

from typing import List, Dict, Optional, Any, Union
from partition_text2gql.parser.tokens import Token, TokenType
from partition_text2gql.parser.lexer import Lexer
from partition_text2gql.parser.ast_nodes import (
    QueryAST, ClauseNode, MatchClause, PatternPath, NodePattern, RelPattern,
    RelDirection, ReturnClause, ReturnItem, OrderByItem, WithClause,
    ExprNode, LiteralExpr, VariableExpr, PropertyAccessExpr, BinaryOpExpr,
    UnaryOpExpr, FunctionCallExpr, ListLiteralExpr, MapLiteralExpr
)
from partition_text2gql.core.errors import CypherSyntaxError


class CypherParser:
    """Recursive-Descent Parser for openCypher/GQL."""

    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = Lexer(source).tokenize()
        self.pos = 0

    @classmethod
    def parse_query(cls, source: str) -> QueryAST:
        """Parse source Cypher string into typed QueryAST."""
        return cls(source).parse()

    def _peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]  # EOF token

    def _match(self, *types: TokenType) -> bool:
        if self._peek().type in types:
            self.pos += 1
            return True
        return False

    def _check(self, *types: TokenType) -> bool:
        return self._peek().type in types

    def _consume(self, token_type: TokenType, err_msg: str) -> Token:
        token = self._peek()
        if token.type == token_type:
            self.pos += 1
            return token
        raise CypherSyntaxError(
            f"{err_msg}. Got '{token.value}' ({token.type.name})",
            line=token.line,
            col=token.col,
            snippet=self.source.splitlines()[token.line - 1] if token.line - 1 < len(self.source.splitlines()) else ""
        )

    # ==========================================
    # Main Query Parser
    # ==========================================

    def parse(self) -> QueryAST:
        clauses: List[ClauseNode] = []
        pending_partition: Optional[int] = None

        while not self._check(TokenType.EOF):
            # Capture partition annotation comment: // partition:N
            if self._check(TokenType.PARTITION_COMMENT):
                tok = self._consume(TokenType.PARTITION_COMMENT, "Expected partition comment")
                try:
                    # Clean up value, might be "1" or "[0, 1]"
                    val_str = str(tok.value).strip().strip("[]")
                    first_val = val_str.split(",")[0].strip()
                    pending_partition = int(first_val)
                except ValueError:
                    pending_partition = None
                continue

            # Strip CALL wrapper if present: CALL { ... }
            if self._check(TokenType.CALL) and self._peek(1).type == TokenType.LBRACE:
                self._consume(TokenType.CALL, "Expected CALL")
                self._consume(TokenType.LBRACE, "Expected '{' after CALL")
                continue

            # Check if closing CALL brace
            if self._check(TokenType.RBRACE):
                self._consume(TokenType.RBRACE, "Expected '}'")
                continue

            # MATCH or OPTIONAL MATCH
            if self._check(TokenType.MATCH) or (self._check(TokenType.OPTIONAL) and self._peek(1).type == TokenType.MATCH):
                clauses.append(self._parse_match_clause(partition_annotation=pending_partition))
                pending_partition = None
                continue

            # WITH clause
            if self._check(TokenType.WITH):
                clauses.append(self._parse_with_clause())
                continue

            # RETURN clause
            if self._check(TokenType.RETURN):
                clauses.append(self._parse_return_clause())
                continue

            # Optional trailing semicolon
            if self._check(TokenType.SEMICOLON):
                self._consume(TokenType.SEMICOLON, "Expected ';'")
                continue

            # Unexpected token
            tok = self._peek()
            raise CypherSyntaxError(f"Unexpected token in query clause: '{tok.value}'", tok.line, tok.col)

        return QueryAST(clauses)

    # ==========================================
    # Clause Parsers
    # ==========================================

    def _parse_match_clause(self, partition_annotation: Optional[int] = None) -> MatchClause:
        optional = False
        if self._match(TokenType.OPTIONAL):
            optional = True
        self._consume(TokenType.MATCH, "Expected 'MATCH'")

        patterns: List[PatternPath] = []
        patterns.append(self._parse_pattern_path())

        while self._match(TokenType.COMMA):
            patterns.append(self._parse_pattern_path())

        where_expr: Optional[ExprNode] = None
        if self._match(TokenType.WHERE):
            where_expr = self._parse_expression()

        return MatchClause(
            patterns=patterns,
            where=where_expr,
            optional=optional,
            partition_annotation=partition_annotation
        )

    def _parse_with_clause(self) -> WithClause:
        self._consume(TokenType.WITH, "Expected 'WITH'")
        distinct = self._match(TokenType.DISTINCT)

        items: List[ReturnItem] = []
        items.append(self._parse_return_item())
        while self._match(TokenType.COMMA):
            items.append(self._parse_return_item())

        where_expr: Optional[ExprNode] = None
        if self._match(TokenType.WHERE):
            where_expr = self._parse_expression()

        return WithClause(items=items, distinct=distinct, where=where_expr)

    def _parse_return_clause(self) -> ReturnClause:
        self._consume(TokenType.RETURN, "Expected 'RETURN'")
        distinct = self._match(TokenType.DISTINCT)

        items: List[ReturnItem] = []
        items.append(self._parse_return_item())
        while self._match(TokenType.COMMA):
            items.append(self._parse_return_item())

        order_by: Optional[List[OrderByItem]] = None
        if self._match(TokenType.ORDER):
            self._consume(TokenType.BY, "Expected 'BY' after 'ORDER'")
            order_by = []
            order_by.append(self._parse_order_by_item())
            while self._match(TokenType.COMMA):
                order_by.append(self._parse_order_by_item())

        skip: Optional[int] = None
        if self._match(TokenType.SKIP):
            skip_tok = self._consume(TokenType.INTEGER_LITERAL, "Expected integer after SKIP")
            skip = int(skip_tok.value)

        limit: Optional[int] = None
        if self._match(TokenType.LIMIT):
            limit_tok = self._consume(TokenType.INTEGER_LITERAL, "Expected integer after LIMIT")
            limit = int(limit_tok.value)

        return ReturnClause(items=items, distinct=distinct, order_by=order_by, skip=skip, limit=limit)

    def _parse_return_item(self) -> ReturnItem:
        expr = self._parse_expression()
        alias: Optional[str] = None
        if self._match(TokenType.AS):
            alias_tok = self._consume(TokenType.IDENTIFIER, "Expected alias identifier after 'AS'")
            alias = alias_tok.value
        return ReturnItem(expression=expr, alias=alias)

    def _parse_order_by_item(self) -> OrderByItem:
        expr = self._parse_expression()
        ascending = True
        if self._match(TokenType.DESC):
            ascending = False
        elif self._match(TokenType.ASC):
            ascending = True
        return OrderByItem(expression=expr, ascending=ascending)

    # ==========================================
    # Graph Pattern Parser
    # ==========================================

    def _parse_pattern_path(self) -> PatternPath:
        elements: List[Union[NodePattern, RelPattern]] = []
        elements.append(self._parse_node_pattern())

        while self._check(TokenType.DASH, TokenType.ARROW_LEFT):
            # RelPattern followed by NodePattern
            elements.append(self._parse_rel_pattern())
            elements.append(self._parse_node_pattern())

        return PatternPath(elements=elements)

    def _parse_node_pattern(self) -> NodePattern:
        self._consume(TokenType.LPAREN, "Expected '(' to begin node pattern")

        var: Optional[str] = None
        labels: List[str] = []
        properties: Dict[str, ExprNode] = {}

        # Variable name
        if self._check(TokenType.IDENTIFIER):
            var = self._consume(TokenType.IDENTIFIER, "").value

        # Labels: :Label1:Label2
        while self._match(TokenType.COLON):
            label_tok = self._consume(TokenType.IDENTIFIER, "Expected label name after ':'")
            labels.append(label_tok.value)

        # Inline properties: { prop: val, ... }
        if self._match(TokenType.LBRACE):
            properties = self._parse_properties_map()
            self._consume(TokenType.RBRACE, "Expected '}' after property map")

        self._consume(TokenType.RPAREN, "Expected ')' to end node pattern")
        return NodePattern(variable=var, labels=labels, properties=properties)

    def _parse_rel_pattern(self) -> RelPattern:
        direction = RelDirection.UNDIRECTED

        # Check left side: <- or -
        if self._match(TokenType.ARROW_LEFT):
            # <-[...] or <--
            direction = RelDirection.INCOMING
            self._consume(TokenType.LBRACKET, "Expected '[' after '<-'")
            rel = self._parse_rel_body(direction)
            self._consume(TokenType.RBRACKET, "Expected ']' after relationship")
            self._consume(TokenType.DASH, "Expected '-' after ']'")
            return rel

        self._consume(TokenType.DASH, "Expected '-'")
        self._consume(TokenType.LBRACKET, "Expected '['")
        rel = self._parse_rel_body(direction)
        self._consume(TokenType.RBRACKET, "Expected ']'")

        # Check right side: -> or -
        if self._match(TokenType.ARROW_RIGHT):
            rel.direction = RelDirection.OUTGOING
        elif self._match(TokenType.DASH):
            rel.direction = RelDirection.UNDIRECTED
        else:
            raise CypherSyntaxError("Expected '->' or '-' after relationship pattern", self._peek().line, self._peek().col)

        return rel

    def _parse_rel_body(self, initial_dir: RelDirection) -> RelPattern:
        var: Optional[str] = None
        rel_types: List[str] = []
        properties: Dict[str, ExprNode] = {}
        varlength: Optional[str] = None

        if self._check(TokenType.IDENTIFIER):
            var = self._consume(TokenType.IDENTIFIER, "").value

        if self._match(TokenType.COLON):
            type_tok = self._consume(TokenType.IDENTIFIER, "Expected relationship type after ':'")
            rel_types.append(type_tok.value)
            while self._match(TokenType.SLASH):  # or |
                next_tok = self._consume(TokenType.IDENTIFIER, "Expected relationship type")
                rel_types.append(next_tok.value)

        # Variable length: *1..3 or *
        if self._match(TokenType.STAR):
            varlength = "*"
            if self._check(TokenType.INTEGER_LITERAL):
                varlength += str(self._consume(TokenType.INTEGER_LITERAL, "").value)
            if self._match(TokenType.DOT):
                self._consume(TokenType.DOT, "Expected second dot in range")
                varlength += ".."
                if self._check(TokenType.INTEGER_LITERAL):
                    varlength += str(self._consume(TokenType.INTEGER_LITERAL, "").value)

        # Inline properties
        if self._match(TokenType.LBRACE):
            properties = self._parse_properties_map()
            self._consume(TokenType.RBRACE, "Expected '}' after property map")

        return RelPattern(variable=var, rel_types=rel_types, properties=properties, direction=initial_dir, varlength=varlength)

    def _parse_properties_map(self) -> Dict[str, ExprNode]:
        props: Dict[str, ExprNode] = {}
        if self._check(TokenType.RBRACE):
            return props

        while True:
            key_tok = self._consume(TokenType.IDENTIFIER, "Expected property key")
            self._consume(TokenType.COLON, "Expected ':' after property key")
            val_expr = self._parse_expression()
            props[key_tok.value] = val_expr
            if not self._match(TokenType.COMMA):
                break
        return props

    # ==========================================
    # Expression Parsing (Precedence Climbing)
    # ==========================================

    def _parse_expression(self) -> ExprNode:
        return self._parse_or()

    def _parse_or(self) -> ExprNode:
        expr = self._parse_and()
        while self._match(TokenType.OR):
            right = self._parse_and()
            expr = BinaryOpExpr(left=expr, op="OR", right=right)
        return expr

    def _parse_and(self) -> ExprNode:
        expr = self._parse_not()
        while self._match(TokenType.AND):
            right = self._parse_not()
            expr = BinaryOpExpr(left=expr, op="AND", right=right)
        return expr

    def _parse_not(self) -> ExprNode:
        if self._match(TokenType.NOT):
            operand = self._parse_not()
            return UnaryOpExpr(op="NOT", operand=operand)
        return self._parse_comparison()

    def _parse_comparison(self) -> ExprNode:
        expr = self._parse_addition()

        comp_ops = {
            TokenType.EQ: "=",
            TokenType.NEQ: "!=",
            TokenType.LT: "<",
            TokenType.LTE: "<=",
            TokenType.GT: ">",
            TokenType.GTE: ">=",
            TokenType.IN: "IN",
            TokenType.CONTAINS: "CONTAINS",
            TokenType.STARTS: "STARTS WITH",
            TokenType.ENDS: "ENDS WITH",
        }

        tok = self._peek()
        if tok.type in comp_ops:
            op_str = comp_ops[tok.type]
            self.pos += 1
            if tok.type in (TokenType.STARTS, TokenType.ENDS):
                self._consume(TokenType.WITH, "Expected 'WITH' after STARTS/ENDS")
            right = self._parse_addition()
            return BinaryOpExpr(left=expr, op=op_str, right=right)

        # IS NULL / IS NOT NULL
        if self._match(TokenType.IS):
            if self._match(TokenType.NOT):
                self._consume(TokenType.NULL, "Expected 'NULL' after 'IS NOT'")
                return BinaryOpExpr(left=expr, op="IS NOT", right=LiteralExpr(None, "null"))
            self._consume(TokenType.NULL, "Expected 'NULL' after 'IS'")
            return BinaryOpExpr(left=expr, op="IS", right=LiteralExpr(None, "null"))

        return expr

    def _parse_addition(self) -> ExprNode:
        expr = self._parse_multiplication()
        while self._check(TokenType.PLUS, TokenType.MINUS):
            op_tok = self._peek()
            self.pos += 1
            right = self._parse_multiplication()
            expr = BinaryOpExpr(left=expr, op=op_tok.value, right=right)
        return expr

    def _parse_multiplication(self) -> ExprNode:
        expr = self._parse_primary()
        while self._check(TokenType.STAR, TokenType.SLASH):
            op_tok = self._peek()
            self.pos += 1
            right = self._parse_primary()
            expr = BinaryOpExpr(left=expr, op=op_tok.value, right=right)
        return expr

    def _parse_primary(self) -> ExprNode:
        # String literal
        if self._check(TokenType.STRING_LITERAL):
            tok = self._consume(TokenType.STRING_LITERAL, "")
            return LiteralExpr(value=tok.value, literal_type="string")

        # Numeric literal
        if self._check(TokenType.INTEGER_LITERAL):
            tok = self._consume(TokenType.INTEGER_LITERAL, "")
            return LiteralExpr(value=tok.value, literal_type="int")
        if self._check(TokenType.FLOAT_LITERAL):
            tok = self._consume(TokenType.FLOAT_LITERAL, "")
            return LiteralExpr(value=tok.value, literal_type="float")

        # Boolean literal
        if self._match(TokenType.TRUE):
            return LiteralExpr(value=True, literal_type="bool")
        if self._match(TokenType.FALSE):
            return LiteralExpr(value=False, literal_type="bool")

        # NULL literal
        if self._match(TokenType.NULL):
            return LiteralExpr(value=None, literal_type="null")

        # Parenthesized expression
        if self._match(TokenType.LPAREN):
            expr = self._parse_expression()
            self._consume(TokenType.RPAREN, "Expected ')'")
            return expr

        # List literal: [1, 2, 3]
        if self._match(TokenType.LBRACKET):
            items: List[ExprNode] = []
            if not self._check(TokenType.RBRACKET):
                items.append(self._parse_expression())
                while self._match(TokenType.COMMA):
                    items.append(self._parse_expression())
            self._consume(TokenType.RBRACKET, "Expected ']'")
            return ListLiteralExpr(items=items)

        # Variable or Function Call or Property Access: ident(...) or ident.prop or ident
        if self._check(TokenType.IDENTIFIER):
            ident = self._consume(TokenType.IDENTIFIER, "").value

            # Function call: func(...)
            if self._match(TokenType.LPAREN):
                is_distinct = self._match(TokenType.DISTINCT)
                args: List[ExprNode] = []
                if not self._check(TokenType.RPAREN):
                    if self._match(TokenType.STAR):
                        args.append(VariableExpr("*"))
                    else:
                        args.append(self._parse_expression())
                    while self._match(TokenType.COMMA):
                        args.append(self._parse_expression())
                self._consume(TokenType.RPAREN, "Expected ')' after function arguments")
                return FunctionCallExpr(function_name=ident, arguments=args, is_distinct=is_distinct)

            # Property access: ident.prop
            if self._match(TokenType.DOT):
                prop_tok = self._consume(TokenType.IDENTIFIER, "Expected property name after '.'")
                return PropertyAccessExpr(variable=ident, property_name=prop_tok.value)

            # Plain variable
            return VariableExpr(name=ident)

        tok = self._peek()
        raise CypherSyntaxError(f"Unexpected token in expression: '{tok.value}'", tok.line, tok.col)
