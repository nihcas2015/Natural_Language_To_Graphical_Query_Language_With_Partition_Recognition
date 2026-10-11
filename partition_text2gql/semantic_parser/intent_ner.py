"""
Intent Classification & Named Entity Recognition for Natural Language Queries (Stage 1.1 & 1.2).
Provides dual-mode architecture:
  - ModelBackedIntentClassifier: Loads fine-tuned DistilBERT multi-task checkpoint when present.
  - HeuristicIntentClassifier: High-precision rule-based matcher for deterministic offline execution.
  - AutoIntentNER: Automatically switches to neural model if checkpoint is available.
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from partition_text2gql.core.types import QueryIntent, EntitySpan


class IntentClassifier:
class HeuristicIntentClassifier:
    """Classifies natural language user intent into structural graph query archetypes."""

    AGG_PATTERNS = [r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\baverage\b", r"\bavg\b", r"\bsum\b", r"\bmaximum\b", r"\bminimum\b"]
    PATH_PATTERNS = [r"\bpath\b", r"\bshortest path\b", r"\broute\b", r"\bconnected to\b", r"\bdegrees of separation\b"]
    MULTI_HOP_PATTERNS = [r"\bwho directed the movie starring\b", r"\bplayers who played for the team coached by\b", r"\bwhere\b.*\bthrough which\b"]
    AGG_PATTERNS = [
        r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\baverage\b", r"\bavg\b",
        r"\bsum\b", r"\bmaximum\b", r"\bminimum\b", r"\bnumber of\b"
    ]
    PATH_PATTERNS = [
        r"\bpath\b", r"\bshortest path\b", r"\broute\b", r"\bconnected to\b",
        r"\bdegrees of separation\b", r"\bconnection between\b"
    ]
    MULTI_HOP_PATTERNS = [
        r"\bwho directed the movie starring\b", r"\bplayers who played for the team coached by\b",
        r"\bwhere\b.*\bthrough which\b", r"\bcompanies based in .* that have board members\b",
        r"\bfriends of friends\b"
    ]

    @classmethod
    def classify(cls, nl_question: str) -> QueryIntent:
        q_lower = nl_question.lower()

        for p in cls.AGG_PATTERNS:
            if re.search(p, q_lower):
                return QueryIntent.AGGREGATION

        for p in cls.PATH_PATTERNS:
            if re.search(p, q_lower):
                return QueryIntent.PATH_TRAVERSAL

        for p in cls.MULTI_HOP_PATTERNS:
            if re.search(p, q_lower):
                return QueryIntent.MULTI_HOP_JOIN

        # Default query intent
        if "what" in q_lower or "which" in q_lower or "find" in q_lower or "who" in q_lower:
        if any(w in q_lower for w in ("what", "which", "find", "who", "give", "list", "show", "provide")):
            return QueryIntent.LOOKUP

        return QueryIntent.SUBGRAPH_EXTRACTION


class ModelBackedIntentClassifier:
    """Loads fine-tuned DistilBERT sequence classifier from Kaggle/local training."""

    INTENT_MAP = {
        0: QueryIntent.LOOKUP,
        1: QueryIntent.PATH_TRAVERSAL,
        2: QueryIntent.AGGREGATION,
        3: QueryIntent.MULTI_HOP_JOIN,
        4: QueryIntent.SUBGRAPH_EXTRACTION
    }

    def __init__(self, model_dir: str = "models/intent_ner_model"):
        self.model_dir = model_dir
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def is_available(self) -> bool:
        return os.path.exists(os.path.join(self.model_dir, "config.json"))

    def _lazy_load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self._model.eval()
        self._loaded = True

    def classify(self, nl_question: str) -> QueryIntent:
        self._lazy_load()
        import torch
        inputs = self._tokenizer(nl_question, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self._model(**inputs)
            pred_id = int(torch.argmax(outputs.logits, dim=-1)[0])
        return self.INTENT_MAP.get(pred_id, QueryIntent.LOOKUP)


class EntityRelationExtractor:
    """Extracts entity literal mentions, property values, and relationship hints from NL prompts."""

    @classmethod
    def extract_entities(cls, nl_question: str) -> List[EntitySpan]:
        spans: List[EntitySpan] = []

        # 1. Quoted literals: '...' or "..."
        for m in re.finditer(r"['\"]([^'\"]+)['\"]", nl_question):
            spans.append(EntitySpan(
                text=m.group(1),
                start_char=m.start(1),
                end_char=m.end(1),
                tag="VALUE",
                confidence=0.98
            ))

        # 2. Year / Integer numbers
        for m in re.finditer(r"\b(19\d\d|20\d\d|\d+)\b", nl_question):
            # Check not already captured
            if not any(s.start_char <= m.start() and m.end() <= s.end_char for s in spans):
                val = int(m.group(1))
                tag = "VALUE_YEAR" if 1900 <= val <= 2100 else "VALUE_INT"
                spans.append(EntitySpan(
                    text=m.group(1),
                    start_char=m.start(),
                    end_char=m.end(),
                    tag=tag,
                    normalized_value=val,
                    confidence=0.90
                ))

        # 3. Capitalized multi-word Named Entities (e.g. Christopher Nolan, Los Angeles Lakers)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", nl_question):
            if not any(s.start_char <= m.start() and m.end() <= s.end_char for s in spans):
                spans.append(EntitySpan(
                    text=m.group(1),
                    start_char=m.start(),
                    end_char=m.end(),
                    tag="ENTITY_NAME",
                    confidence=0.85
                ))

        return spans


# Legacy alias
IntentClassifier = HeuristicIntentClassifier


class AutoIntentNER:
    """Unified access layer auto-selecting neural model or heuristic fallback."""

    def __init__(self, model_dir: str = "models/intent_ner_model"):
        self.neural = ModelBackedIntentClassifier(model_dir)

    def classify_intent(self, nl_question: str) -> QueryIntent:
        if self.neural.is_available():
            try:
                return self.neural.classify(nl_question)
            except Exception:
                pass
        return HeuristicIntentClassifier.classify(nl_question)

    def extract_entities(self, nl_question: str) -> List[EntitySpan]:
        return EntityRelationExtractor.extract_entities(nl_question)
