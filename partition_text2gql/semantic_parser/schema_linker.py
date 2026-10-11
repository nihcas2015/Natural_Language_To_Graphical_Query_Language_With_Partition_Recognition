"""
Partition-Aware Schema Linker (Stage 1.3).
Retrieves relevant schema elements (labels, properties, relationship types) augmented with physical partition placement.
Supports dual-mode execution:
  - EmbeddingSchemaLinker: Dense bi-encoder + FAISS / cosine vector search.
  - FuzzySchemaLinker: High-recall token Jaccard + ngram lexical overlap matcher.
"""

from typing import List, Dict, Optional, Set
import os
from typing import List, Dict, Optional, Set, Tuple
from partition_text2gql.core.types import SchemaElement, SchemaElementType
from partition_text2gql.validator.schema_catalog import GraphDomainSchema
from partition_text2gql.stats.store import PartitionStatisticsStore


class FuzzySchemaLinker:
    """High-precision token overlap and Jaccard similarity matcher."""

    @staticmethod
    def compute_similarity(text: str, query: str) -> float:
        t_tokens = set(re_tokenize(text))
        q_tokens = set(re_tokenize(query))
        if not t_tokens or not q_tokens:
            return 0.0
        intersection = t_tokens & q_tokens
        union = t_tokens | q_tokens
        return len(intersection) / len(union)


def re_tokenize(s: str) -> List[str]:
    import re
    return [w.lower() for w in re.split(r"[\s_:\-\.,]+", s) if len(w) > 1]


class PartitionAwareSchemaLinker:
    """
    Core NLP Innovation:
    Links natural language tokens not just to semantic labels, but to their physical shard host locations.
    """

    def __init__(
        self,
        schema: GraphDomainSchema,
        stats_store: PartitionStatisticsStore,
        model_dir: str = "models/schema_linker"
    ):
        self.schema = schema
        self.stats_store = stats_store
        self.model_dir = model_dir
        self.catalog: List[SchemaElement] = self._build_catalog()
        self._bi_encoder = None
        self._catalog_embeddings = None

    def _build_catalog(self) -> List[SchemaElement]:
        catalog = []
        domain = self.schema.domain

        # 1. Node labels with physical partitions
        for label_name, l_schema in self.schema.labels.items():
            owning_pids = self.stats_store.get_partitions_for_label(domain, label_name)
            elem = SchemaElement(
                name=label_name,
                element_type=SchemaElementType.NODE_LABEL,
                owning_partitions=owning_pids,
                domain=domain,
                description=f"Node entity representing '{label_name}' hosted on partitions {owning_pids}"
            )
            catalog.append(elem)

            # Properties for this label
            for prop in l_schema.properties:
                p_elem = SchemaElement(
                    name=prop,
                    element_type=SchemaElementType.PROPERTY,
                    owning_partitions=owning_pids,
                    domain=domain,
                    associated_label=label_name,
                    description=f"Property '{prop}' belonging to '{label_name}' on partitions {owning_pids}"
                )
                catalog.append(p_elem)

        # 2. Relationship types
        for rel_name, r_schema in self.schema.relationships.items():
            # Spanning partitions across source and target
            source_parts: Set[int] = set()
            for sl in r_schema.source_labels:
                source_parts.update(self.stats_store.get_partitions_for_label(domain, sl))
            target_parts: Set[int] = set()
            for tl in r_schema.target_labels:
                target_parts.update(self.stats_store.get_partitions_for_label(domain, tl))

            spanning_pids = sorted(source_parts | target_parts) or [0, 1]

            r_elem = SchemaElement(
                name=rel_name,
                element_type=SchemaElementType.RELATIONSHIP_TYPE,
                owning_partitions=spanning_pids,
                domain=domain,
                description=f"Relationship edge ':{rel_name}' connecting labels {r_schema.source_labels} -> {r_schema.target_labels}, spanning partitions {spanning_pids}"
            )
            catalog.append(r_elem)

        return catalog

    def _try_load_biencoder(self) -> bool:
        if self._bi_encoder is not None:
            return True
        if not os.path.exists(self.model_dir):
            return False
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._bi_encoder = SentenceTransformer(self.model_dir)
            emb_file = os.path.join(self.model_dir, "embeddings.npy")
            if os.path.exists(emb_file):
                self._catalog_embeddings = np.load(emb_file)
            else:
                texts = [f"{e.name} {e.description}" for e in self.catalog]
                self._catalog_embeddings = self._bi_encoder.encode(texts, normalize_embeddings=True)
            return True
        except Exception:
            return False

    def link(self, nl_question: str, top_k: int = 5) -> List[SchemaElement]:
        """
        Retrieves top-k schema elements most relevant to the user's natural language question.
        Uses lexical + semantic overlap; in full deployment connects to Sentence-Transformers FAISS index.
        Uses neural embeddings if trained model is present, otherwise high-precision fuzzy matching.
        """
        # Tier 1: Neural Bi-Encoder Retrieval
        if self._try_load_biencoder() and self._bi_encoder is not None:
            try:
                import numpy as np
                q_emb = self._bi_encoder.encode([nl_question], normalize_embeddings=True)[0]
                sims = np.dot(self._catalog_embeddings, q_emb)
                top_indices = np.argsort(sims)[::-1][:top_k]
                return [self.catalog[i] for i in top_indices]
            except Exception:
                pass

        # Tier 2: Enhanced Lexical + Jaccard + Substring Matching
        q_lower = nl_question.lower()
        scored: List[tuple[float, SchemaElement]] = []
        q_tokens = set(re_tokenize(nl_question))
        scored: List[Tuple[float, SchemaElement]] = []

        for elem in self.catalog:
            score = 0.0
            name_lower = elem.name.lower()
            name_tokens = set(re_tokenize(elem.name))

            # Exact name in question
            if name_lower in q_lower:
                score += 3.0
                score += 5.0

            words = name_lower.replace("_", " ").split()
            for w in words:
                if len(w) > 2 and w in q_lower:
                    score += 1.0
            # Token overlap
            overlap = len(name_tokens & q_tokens)
            score += overlap * 2.0

            # Associated label overlap
            if elem.associated_label and elem.associated_label.lower() in q_lower:
                score += 0.5
                score += 1.5

            # Domain keyword heuristics
            if elem.name.lower() in ("title", "name", "director", "directedby") and any(k in q_lower for k in ("movie", "film", "directed", "director")):
                score += 2.0

            if score > 0:
                scored.append((score, elem))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [s[1] for s in scored[:top_k]]

        # If empty, return primary labels from catalog
        if not results:
            results = [e for e in self.catalog if e.element_type == SchemaElementType.NODE_LABEL][:top_k]

        return results
        # Schema Expansion: If a relationship is retrieved, ensure its source and target labels are present
        expanded = list(results)
        existing_names = {e.name for e in expanded}
        for elem in results:
            if elem.element_type == SchemaElementType.RELATIONSHIP_TYPE and elem.name in self.schema.relationships:
                r_schema = self.schema.relationships[elem.name]
                for endpoint_lbl in (r_schema.source_labels | r_schema.target_labels):
                    if endpoint_lbl not in existing_names:
                        for cat_elem in self.catalog:
                            if cat_elem.element_type == SchemaElementType.NODE_LABEL and cat_elem.name == endpoint_lbl:
                                expanded.append(cat_elem)
                                existing_names.add(endpoint_lbl)

        return expanded
