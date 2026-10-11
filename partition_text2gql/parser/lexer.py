"""
Formal Lexical Analyzer (Scanner) for openCypher / GQL.
No regex hacks: deterministic stateful character-by-character scanner with precise line & column tracking.
"""

from typing import List, Optional
from partition_text2gql.parser.tokens import Token, TokenType, KEYWORDS
from partition_text2gql.core.errors import CypherSyntaxError


class Lexer:
    """Stateful Lexer for openCypher."""

    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.col = 1

    def _peek(self, offset: int = 0) -> Optional[str]:
        target = self.pos + offset
        if target < self.length:
            return self.source[target]
        return None

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.pos < self.length:
            ch = self._peek()

            # 1. Skip whitespace
            if ch in (' ', '\t', '\r', '\n'):
                self._advance()
                continue

            # 2. Comments: single-line // ...
            if ch == '/' and self._peek(1) == '/':
                start_line, start_col = self.line, self.col
                self._advance()  # /
                self._advance()  # /
                comment_text = ""
                while self.pos < self.length and self._peek() != '\n':
                    comment_text += self._advance()
                
                comment_stripped = comment_text.strip()
                # Check for partition annotation comment: // partition:N or // partitions: [0, 1]
                if comment_stripped.lower().startswith("partition:") or comment_stripped.lower().startswith("partitions:"):
                    parts = comment_stripped.split(":", 1)
                    val = parts[1].strip()
                    tokens.append(Token(TokenType.PARTITION_COMMENT, val, start_line, start_col))
                continue

            # 3. String literals ('...' or "...")
            if ch in ("'", '"'):
                tokens.append(self._scan_string(ch))
                continue

            # 4. Arrows and dashes: <-, ->, -, <=, <>
            if ch == '<':
                start_line, start_col = self.line, self.col
                self._advance()
                if self._peek() == '-':
                    self._advance()
                    tokens.append(Token(TokenType.ARROW_LEFT, "<-", start_line, start_col))
                elif self._peek() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.LTE, "<=", start_line, start_col))
                elif self._peek() == '>':
                    self._advance()
                    tokens.append(Token(TokenType.NEQ, "<>", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.LT, "<", start_line, start_col))
                continue

            if ch == '-':
                start_line, start_col = self.line, self.col
                self._advance()
                if self._peek() == '>':
                    self._advance()
                    tokens.append(Token(TokenType.ARROW_RIGHT, "->", start_line, start_col))
                elif self._peek() and self._peek().isdigit():
                    # Could be negative number if preceded by operator, handled by parser or number scan
                    tokens.append(Token(TokenType.MINUS, "-", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.DASH, "-", start_line, start_col))
                continue

            # 5. Equality & inequality: =, !=
            if ch == '=':
                start_line, start_col = self.line, self.col
                self._advance()
                tokens.append(Token(TokenType.EQ, "=", start_line, start_col))
                continue

            if ch == '!' and self._peek(1) == '=':
                start_line, start_col = self.line, self.col
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.NEQ, "!=", start_line, start_col))
                continue

            if ch == '>':
                start_line, start_col = self.line, self.col
                self._advance()
                if self._peek() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.GTE, ">=", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.GT, ">", start_line, start_col))
                continue

            # 6. Single-character punctuation
            single_char_map = {
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                ':': TokenType.COLON,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
                ';': TokenType.SEMICOLON,
                '*': TokenType.STAR,
                '+': TokenType.PLUS,
                '/': TokenType.SLASH,
            }
            if ch in single_char_map:
                start_line, start_col = self.line, self.col
                tokens.append(Token(single_char_map[ch], self._advance(), start_line, start_col))
                continue

            # 7. Numeric literals
            if ch.isdigit():
                tokens.append(self._scan_number())
                continue

            # 8. Identifiers & Keywords
            if ch.isalpha() or ch == '_':
                tokens.append(self._scan_identifier())
                continue

            # Unknown character
            bad_ch = self._advance()
            raise CypherSyntaxError(f"Unexpected character: '{bad_ch}'", self.line, self.col)

        tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return tokens

    def _scan_string(self, quote: str) -> Token:
        start_line, start_col = self.line, self.col
        self._advance()  # consume opening quote
        chars: List[str] = []

        while self.pos < self.length:
            ch = self._advance()
            if ch == quote:
                return Token(TokenType.STRING_LITERAL, "".join(chars), start_line, start_col)
            if ch == '\\':
                if self.pos < self.length:
                    escaped = self._advance()
                    escape_map = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', quote: quote}
                    chars.append(escape_map.get(escaped, escaped))
                else:
                    break
            else:
                chars.append(ch)

        raise CypherSyntaxError(f"Unterminated string literal starting at line {start_line}, col {start_col}", start_line, start_col)

    def _scan_number(self) -> Token:
        start_line, start_col = self.line, self.col
        num_str = ""
        is_float = False

        while self.pos < self.length and self._peek().isdigit():
            num_str += self._advance()

        if self._peek() == '.' and self._peek(1) and self._peek(1).isdigit():
            is_float = True
            num_str += self._advance()  # .
            while self.pos < self.length and self._peek().isdigit():
                num_str += self._advance()

        if is_float:
            return Token(TokenType.FLOAT_LITERAL, float(num_str), start_line, start_col)
        return Token(TokenType.INTEGER_LITERAL, int(num_str), start_line, start_col)

    def _scan_identifier(self) -> Token:
        start_line, start_col = self.line, self.col
        ident = ""

        while self.pos < self.length:
            ch = self._peek()
            if ch.isalnum() or ch == '_':
                ident += self._advance()
            else:
                break

        upper = ident.upper()
        if upper in KEYWORDS:
            return Token(KEYWORDS[upper], ident, start_line, start_col)
        return Token(TokenType.IDENTIFIER, ident, start_line, start_col)

