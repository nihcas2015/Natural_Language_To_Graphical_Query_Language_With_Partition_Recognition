"""
Parse Cypher queries to extract node labels and variable bindings from MATCH clauses.

Uses a custom tokenizer (regex-based) since cypher-linter and antlr4-opencypher
are not readily available as pip packages on this environment.

Supports standard Cypher syntax like:
  MATCH (n:Label {prop: val})-[:REL]->(m:Label2)
  MATCH (n:Label)-[r:REL]->(m:Label2)
  MATCH (n:Label)
"""

import re
from typing import List, Dict, Tuple, Set, Optional


# Regex patterns for Cypher parsing
NODE_PATTERN = re.compile(
    r'\(([^)]+)\)',  # Match parenthesized nodes: (n:Label {props})
    re.VERBOSE
)
REL_PATTERN = re.compile(
    r'\[([^\]]*)\]',  # Match bracketed relations: [r:LABEL {props}]
)
LABEL_PATTERN = re.compile(
    r':([A-Z][a-zA-Z0-9_]*)',  # Match labels after colon
)
VAR_PATTERN = re.compile(
    r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::|[:\s{,]|$)',
    # Match variable names before : or space or {
)
PROPERTY_PATTERN = re.compile(
    r'(\w+)\s*:\s*([^\s,}\]]+)',  # Match property: value pairs
)


def parse_match_clause(cypher_query: str) -> Dict:
    """
    Parse the MATCH clause of a Cypher query.

    Returns
    -------
    dict : {
        'node_patterns': list of {var, label, properties},
        'rel_patterns': list of {var, label, source, target},
        'variable_labels': {var: label},
        'referenced_labels': set of label strings
    }
    """
    # Find the MATCH clause (everything between MATCH and WITH/RETURN/WHERE)
    match_match = re.search(
        r'MATCH\s+(.*?)(?:\s+(?:WITH|RETURN|WHERE|ORDER|LIMIT|SKIP|UNWIND|CREATE|MERGE|DELETE|SET|FOREACH|CALL)\s|$)',
        cypher_query, re.IGNORECASE | re.DOTALL
    )
    if not match_match:
        return {'node_patterns': [], 'rel_patterns': [], 'variable_labels': {}, 'referenced_labels': set()}

    match_body = match_match.group(1)

    # Extract node patterns
    node_patterns = []
    variable_labels = {}
    referenced_labels = set()

    # Find all node patterns: (var:Label {props}) or (var:Label) or (var) or (var:Label {prop: val})
    for node_match in NODE_PATTERN.finditer(match_body):
        node_str = node_match.group(1).strip()
        # Parse variable and optional label
        var = None
        label = None
        properties = {}

        parts = re.split(r'\s*:\s*(?=[A-Z])', node_str, maxsplit=1)
        if parts:
            var = parts[0].strip()
            if len(parts) > 1:
                rest = parts[1].strip()
                label_match = re.match(r'([A-Z][a-zA-Z0-9_]*)', rest)
                if label_match:
                    label = label_match.group(1)
                    rest = rest[label_match.end():].strip()
                # Parse properties if present
                if rest.startswith('{') and rest.endswith('}'):
                    props_str = rest[1:-1]
                    properties = parse_properties(props_str)

        if var:
            node_patterns.append({'var': var, 'label': label, 'properties': properties})
            variable_labels[var] = label
            if label:
                referenced_labels.add(label)

    # Extract relationship patterns and their source/target node variables
    rel_patterns = []
    for rel_match in REL_PATTERN.finditer(match_body):
        rel_str = rel_match.group(1).strip()
        # Parse: r:LABEL or r or r:LABEL {props}
        var = None
        label = None
        properties = {}

        parts = re.split(r'\s*:\s*(?=[A-Z])', rel_str, maxsplit=1)
        parts = re.split(r'\s*:\s*(?=[a-zA-Z])', rel_str, maxsplit=1)
        if parts:
            var = parts[0].strip()
            if len(parts) > 1:
                rest = parts[1].strip()
                label_match = re.match(r'([A-Z][a-zA-Z0-9_]*)', rest)
                label_match = re.match(r'([a-zA-Z][a-zA-Z0-9_]*)', rest)
                if label_match:
                    label = label_match.group(1)
                    rest = rest[label_match.end():].strip()
                if rest.startswith('{') and rest.endswith('}'):
                    properties = parse_properties(rest[1:-1])

        if var:
            rel_patterns.append({'var': var, 'label': label, 'properties': properties})

    return {
        'node_patterns': node_patterns,
        'rel_patterns': rel_patterns,
        'variable_labels': variable_labels,
        'referenced_labels': sorted(referenced_labels)
    }


def parse_properties(props_str: str) -> Dict:
    """Parse a property string like 'name: \"foo\", age: 5' into a dict."""
    props = {}
    # Simple key: value pairs
    for m in re.finditer(r'(\w+)\s*:\s*([^,}]+)', props_str):
        key = m.group(1)
        val = m.group(2).strip()
        # Try to parse as string or number
        if val.startswith("'") and val.endswith("'"):
            val = val[1:-1]
        elif val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        else:
            try:
                if '.' in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass
        props[key] = val
    return props


def extract_labels_from_cypher(cypher_query: str) -> List[str]:
    """Extract all node labels referenced in the MATCH clause."""
    parsed = parse_match_clause(cypher_query)
    return parsed['referenced_labels']


def extract_variable_bindings(cypher_query: str) -> Dict[str, Optional[str]]:
    """Return {variable: label} for all node variables in the MATCH clause."""
    parsed = parse_match_clause(cypher_query)
    return parsed['variable_labels']


if __name__ == "__main__":
    test_queries = [
        "MATCH (n:Movie)-[r:directedBy]->(m:Person) RETURN n.name",
        "MATCH (n:Player)-[:playsFor]->(m:Team) RETURN n.name",
        "MATCH (n:Company)-[:basedIn]->(m:Country) RETURN n.name",
        "MATCH (n) WHERE n.name = 'foo' RETURN n",
    ]
    for q in test_queries:
        result = parse_match_clause(q)
        print(f"Query: {q[:60]}...")
        print(f"  Labels: {result['referenced_labels']}")
        print(f"  Vars: {result['variable_labels']}")
        print()
