"""Domain-specific exception hierarchy for Partition-Aware Text2GQL."""

from typing import Optional, Any, Dict


class Text2GQLError(Exception):
    """Base exception for all errors in the partition-aware text-to-GQL pipeline."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CypherSyntaxError(Text2GQLError):
    """Raised when query violates openCypher/GQL formal grammar rules."""
    def __init__(self, message: str, line: int = 1, col: int = 1, snippet: Optional[str] = None):
        details = {"line": line, "col": col, "snippet": snippet}
        formatted = f"Syntax Error at line {line}, col {col}: {message}"
        if snippet:
            formatted += f"\n  --> {snippet}"
        super().__init__(formatted, details)
        self.line = line
        self.col = col
        self.snippet = snippet


class SchemaValidationError(Text2GQLError):
    """Raised when query references non-existent node labels, edge types, or properties."""
    def __init__(self, message: str, element_type: str, element_name: str, domain: Optional[str] = None):
        details = {"element_type": element_type, "element_name": element_name, "domain": domain}
        formatted = f"Schema Conformance Violation: {element_type} '{element_name}' does not exist in schema."
        if domain:
            formatted += f" (Domain: {domain})"
        super().__init__(formatted, details)
        self.element_type = element_type
        self.element_name = element_name


class SecurityViolationError(Text2GQLError):
    """Raised when query violates safety rules (e.g., destructive operations, unbounded traversals)."""
    def __init__(self, message: str, violation_code: str):
        details = {"violation_code": violation_code}
        super().__init__(f"Security Policy Violation [{violation_code}]: {message}", details)
        self.violation_code = violation_code


class QueryPlanningError(Text2GQLError):
    """Raised when partition decomposition or plan enumeration fails."""
    pass


class ExecutionTimeoutError(Text2GQLError):
    """Raised when a shard subquery exceeds the sandboxed execution threshold."""
    def __init__(self, partition_id: int, timeout_sec: float):
        details = {"partition_id": partition_id, "timeout_sec": timeout_sec}
        super().__init__(f"Shard {partition_id} execution timed out after {timeout_sec:.2f} seconds.", details)
        self.partition_id = partition_id
        self.timeout_sec = timeout_sec

