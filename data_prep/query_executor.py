"""
Query execution and node resolution for gold Cypher queries.

Executes gold Cypher queries against the NetworkX graph to find which
physical nodes they touch, then maps those nodes to partition IDs.
"""

import networkx as nx
from collections import Counter, defaultdict
from typing import List, Dict, Set
from cypher_parser import parse_match_clause


class QueryExecutor:
    """Batch query executor with pre-built label indices."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self._label_index = self._build_label_index()

    def _build_label_index(self) -> Dict[str, Set[str]]:
        """Build {label: set_of_node_ids} index once."""
        index = defaultdict(set)
        for node_id, attrs in self.graph.nodes(data=True):
            label = attrs.get('label')
            if label:
                index[label].add(node_id)
        return dict(index)

    def execute(self, query: str) -> Set[str]:
        """Execute a Cypher query and return touched node IDs."""
        parsed = parse_match_clause(query)
        node_patterns = parsed['node_patterns']
        variable_labels = parsed['variable_labels']

        if not node_patterns:
            labels = set(variable_labels.values()) - {None}
            result = set()
            for label in labels:
                result.update(self._label_index.get(label, set()))
            return result

        # Get candidates for each variable
        var_to_candidates = {}
        for np in node_patterns:
            var = np['var']
            label = np['label']
            if label:
                candidates = set(self._label_index.get(label, set()))
                if np['properties']:
                    filtered = set()
                    for nid in candidates:
                        attrs = self.graph.nodes[nid]
                        if all(attrs.get(pk) == pv for pk, pv in np['properties'].items()):
                            filtered.add(nid)
                    candidates = filtered
                var_to_candidates[var] = candidates
            else:
                var_to_candidates[var] = set(self.graph.nodes())

        if len(node_patterns) == 1:
            var = node_patterns[0]['var']
            return var_to_candidates.get(var, set())
        elif len(node_patterns) == 2:
            var1 = node_patterns[0]['var']
            var2 = node_patterns[1]['var']
            cands1 = var_to_candidates.get(var1, set())
            cands2 = var_to_candidates.get(var2, set())
            if not cands1 or not cands2:
                return cands1 | cands2
            # Find edges between candidate sets
            result = set()
            # Iterate smaller set for efficiency
            if len(cands1) > len(cands2):
                cands1, cands2 = cands2, cands1
            for n1 in cands1:
                for n2 in self.graph.neighbors(n1):
                    if n2 in cands2:
                        result.add(n1)
                        result.add(n2)
            if not result:
                result = cands1 | cands2
            return result
        else:
            result = set()
            for var, cands in var_to_candidates.items():
                result.update(cands)
            return result


# Module-level convenience functions
_query_executors = {}

def _get_executor(graph: nx.DiGraph) -> QueryExecutor:
    """Get or create a QueryExecutor for a graph."""
    gid = id(graph)
    if gid not in _query_executors:
        _query_executors[gid] = QueryExecutor(graph)
    return _query_executors[gid]


def execute_cypher_query(graph: nx.DiGraph, query: str) -> Set[str]:
    """Execute a gold Cypher query against the NetworkX graph."""
    executor = _get_executor(graph)
    return executor.execute(query)


def get_partition_ids(touched_nodes: Set[str], partition_map: Dict[str, int]) -> List[int]:
    """Get sorted partition IDs for a set of touched nodes."""
    part_ids = set()
    for node_id in touched_nodes:
        if node_id in partition_map:
            part_ids.add(partition_map[node_id])
    return sorted(part_ids)


def is_cross_partition(touched_nodes: Set[str], partition_map: Dict[str, int]) -> bool:
    """Return True if touched nodes span more than 1 partition."""
    return len(get_partition_ids(touched_nodes, partition_map)) > 1
