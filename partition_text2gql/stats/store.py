"""
Partition Statistics Store (Persistent).
Stores physical graph partition topologies, boundary cut matrices, vertex-to-shard indices, and degree histograms.
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from partition_text2gql.core.types import PartitionTopology


class PartitionStatisticsStore:
    """
    Persistent repository of distributed graph topology and partition statistics.
    Used by Path B (Cost-Based Optimizer) to evaluate hop costs, shuffle volumes, and subquery placement.
    """

    def __init__(self, storage_dir: str = "data/maps"):
        self.storage_dir = storage_dir
        self.topologies: Dict[str, PartitionTopology] = {}
        self.partition_maps: Dict[str, Dict[str, int]] = {}  # {domain: {node_id: partition_id}}
        os.makedirs(storage_dir, exist_ok=True)

    def register_domain_partitioning(
        self,
        domain: str,
        partition_map: Dict[str, int],
        num_partitions: int = 4,
        edge_list: Optional[List[Tuple[str, str]]] = None,
        node_labels: Optional[Dict[str, str]] = None
    ) -> PartitionTopology:
        """Computes and records full physical partition statistics for a graph domain."""
        self.partition_maps[domain] = partition_map

        # 1. Compute partition sizes
        sizes: Dict[int, int] = {p: 0 for p in range(num_partitions)}
        for node, pid in partition_map.items():
            sizes[pid] = sizes.get(pid, 0) + 1

        # 2. Compute edge cut and boundary matrix
        boundary_edges: Dict[str, int] = {}
        total_edges = 0
        edge_cut = 0

        if edge_list:
            total_edges = len(edge_list)
            for u, v in edge_list:
                pu = partition_map.get(u, 0)
                pv = partition_map.get(v, 0)
                if pu != pv:
                    edge_cut += 1
                    key = f"{min(pu, pv)}<->{max(pu, pv)}"
                    boundary_edges[key] = boundary_edges.get(key, 0) + 1

        cut_ratio = (edge_cut / total_edges) if total_edges > 0 else 0.0

        # 3. Compute primary labels per partition
        primary_labels: Dict[int, List[str]] = {p: [] for p in range(num_partitions)}
        if node_labels:
            label_counts: Dict[int, Dict[str, int]] = {p: {} for p in range(num_partitions)}
            for node, pid in partition_map.items():
                lbl = node_labels.get(node)
                if lbl:
                    label_counts[pid][lbl] = label_counts[pid].get(lbl, 0) + 1
            for p, counts in label_counts.items():
                # Top labels for this partition
                sorted_lbls = sorted(counts.keys(), key=lambda l: counts[l], reverse=True)
                primary_labels[p] = sorted_lbls[:5]

        topology = PartitionTopology(
            domain=domain,
            num_partitions=num_partitions,
            partition_sizes=sizes,
            edge_cut_ratio=round(cut_ratio, 4),
            boundary_edges=boundary_edges,
            primary_labels_per_partition=primary_labels
        )
        self.topologies[domain] = topology
        return topology

    def get_node_partition(self, domain: str, node_id: str) -> Optional[int]:
        """Resolves the physical partition hosting a specific node instance."""
        domain_map = self.partition_maps.get(domain)
        if domain_map:
            return domain_map.get(node_id)
        return None

    def get_topology(self, domain: str) -> Optional[PartitionTopology]:
        return self.topologies.get(domain)

    def get_partitions_for_label(self, domain: str, label: str) -> List[int]:
        """Returns all partition IDs containing instances of the given label."""
        top = self.get_topology(domain)
        if not top:
            return list(range(4))
        matches = []
        for pid, labels in top.primary_labels_per_partition.items():
            if label in labels:
                matches.append(pid)
        return matches or list(range(top.num_partitions))

    def save_to_disk(self, domain: str) -> str:
        """Persists topology metadata to disk."""
        top = self.topologies.get(domain)
        if not top:
            raise ValueError(f"No topology registered for domain '{domain}'")
        filepath = os.path.join(self.storage_dir, f"{domain}_topology.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(top.model_dump_json(indent=2))
        return filepath

    def load_from_disk(self, domain: str) -> Optional[PartitionTopology]:
        """Loads persisted topology metadata from disk."""
        filepath = os.path.join(self.storage_dir, f"{domain}_topology.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                top = PartitionTopology(**data)
                self.topologies[domain] = top
                return top
        return None

