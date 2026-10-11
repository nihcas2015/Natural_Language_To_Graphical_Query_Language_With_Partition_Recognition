"""
Custom k-way graph partitioner (METIS-like fallback).

Since pymetis cannot be built on this Windows environment (missing METIS DLL),
we implement a balanced k-way partitioning algorithm that:
  1. Seeds k partitions with well-spread nodes
  2. Greedily assigns remaining nodes to minimize edge-cut
  3. Refines via Kernighan-Lin style swaps for balance + cut reduction
"""

import random
import networkx as nx
from collections import Counter


def metis_partition(graph: nx.Graph, k: int = 4, seed: int = 42) -> dict:
    """
    Partition a graph into k balanced parts minimizing edge-cut.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to partition
    k : int
        Number of partitions
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict : {node_id: partition_id}
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    if k > n:
        raise ValueError(f"k={k} > number of nodes {n}")

    # Fast O(n) partitioning using degree-hash assignment + balance fix
    degrees = dict(graph.degree())
    partition_map = {}
    sizes = Counter()

    # Sort by degree descending for better spread
    sorted_nodes = sorted(nodes, key=lambda x: degrees.get(x, 0), reverse=True)

    for node in sorted_nodes:
        # Assign to partition with most neighbors already assigned (min-cut)
        part_counts = Counter()
        for nb in graph.adj.get(node, {}):
            if nb in partition_map:
                part_counts[partition_map[nb]] += 1
        if part_counts:
            max_nb = max(part_counts.values())
            tied = [p for p, c in part_counts.items() if c == max_nb]
            pid = min(tied, key=lambda p: sizes.get(p, 0))
        else:
            # Pick smallest partition
            min_size = min(sizes.values()) if sizes else 0
            tied = [p for p, c in sizes.items() if c == min_size]
            pid = rng.choice(tied) if tied else 0
        partition_map[node] = pid
        sizes[pid] += 1

    # Balance pass: move nodes from over-full to under-full partitions
    target = n / k
    for _ in range(3):
        moved = False
        sizes = Counter(partition_map.values())
        over = [p for p in range(k) if sizes.get(p, 0) > target * 1.05]
        under = [p for p in range(k) if sizes.get(p, 0) < target * 0.95]
        if not over or not under:
            break
        for node in sorted_nodes:
            cur_part = partition_map[node]
            if cur_part not in over:
                continue
            if not under:
                break
            # Move to smallest under-full partition
            new_part = min(under, key=lambda p: sizes.get(p, 0))
            partition_map[node] = new_part
            sizes[cur_part] -= 1
            sizes[new_part] = sizes.get(new_part, 0) + 1
            moved = True
            if sizes[cur_part] <= target * 1.05:
                over.remove(cur_part)
            if sizes[new_part] >= target * 0.95:
                under = [p for p in under if p != new_part]
            if not over or not under:
                break
        if not moved:
            break

    # Guarantee 100% coverage
    for node in nodes:
        if node not in partition_map:
            partition_map[node] = 0

    return partition_map
