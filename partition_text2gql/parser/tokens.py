"""
Formal Lexical Token definitions for openCypher / GQL.
"""

from enum import Enum, auto
from typing import Any, NamedTuple


class TokenType(Enum):
    # Keywords
    MATCH = auto()
    OPTIONAL = auto()
    WHERE = auto()
    WITH = auto()
    RETURN = auto()
    ORDER = auto()
    BY = auto()
    ASC = auto()
    DESC = auto()
    SKIP = auto()
    LIMIT = auto()
    DISTINCT = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IN = auto()
    IS = auto()
    NULL = auto()
    TRUE = auto()
    FALSE = auto()
    CALL = auto()
    AS = auto()
    CONTAINS = auto()
    STARTS = auto()
    ENDS = auto()

    # Forbidden / Safety Keywords (tracked to reject early)
    CREATE = auto()
    MERGE = auto()
    SET = auto()
    DELETE = auto()
    REMOVE = auto()
    DROP = auto()
    DETACH = auto()

    # Literals & Identifiers
    IDENTIFIER = auto()
    STRING_LITERAL = auto()
    INTEGER_LITERAL = auto()
    FLOAT_LITERAL = auto()

    # Punctuation & Delimiters
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    LBRACKET = auto()     # [
    RBRACKET = auto()     # ]
    LBRACE = auto()       # {
    RBRACE = auto()       # }
    COLON = auto()        # :
    COMMA = auto()        # ,
    DOT = auto()          # .
    SEMICOLON = auto()    # ;
    STAR = auto()         # *

    # Operators
    EQ = auto()           # =
    NEQ = auto()          # != or <>
    LT = auto()           # <
    LTE = auto()          # <=
    GT = auto()           # >
    GTE = auto()          # >=
    PLUS = auto()         # +
    MINUS = auto()        # -
    SLASH = auto()        # /

    # Graph Traversal Arrows
    DASH = auto()         # -
    ARROW_RIGHT = auto()  # ->
    ARROW_LEFT = auto()   # <-

    # Metadata Comments (e.g. // partition:N)
    PARTITION_COMMENT = auto()

    # Stream Termination
    EOF = auto()


KEYWORDS = {
    "MATCH": TokenType.MATCH,
    "OPTIONAL": TokenType.OPTIONAL,
    "WHERE": TokenType.WHERE,
    "WITH": TokenType.WITH,
    "RETURN": TokenType.RETURN,
    "ORDER": TokenType.ORDER,
    "BY": TokenType.BY,
    "ASC": TokenType.ASC,
    "DESC": TokenType.DESC,
    "SKIP": TokenType.SKIP,
    "LIMIT": TokenType.LIMIT,
    "DISTINCT": TokenType.DISTINCT,
    "AND": TokenType.AND,
    "OR": TokenType.OR,
    "NOT": TokenType.NOT,
    "IN": TokenType.IN,
    "IS": TokenType.IS,
    "NULL": TokenType.NULL,
    "TRUE": TokenType.TRUE,
    "FALSE": TokenType.FALSE,
    "CALL": TokenType.CALL,
    "AS": TokenType.AS,
    "CONTAINS": TokenType.CONTAINS,
    "STARTS": TokenType.STARTS,
    "ENDS": TokenType.ENDS,
    "CREATE": TokenType.CREATE,
    "MERGE": TokenType.MERGE,
    "SET": TokenType.SET,
    "DELETE": TokenType.DELETE,
    "REMOVE": TokenType.REMOVE,
    "DROP": TokenType.DROP,
    "DETACH": TokenType.DETACH,
}


class Token(NamedTuple):
    type: TokenType
    value: Any
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {repr(self.value)}, L{self.line}:C{self.col})"

