"""
Prompt Builder for Partition-Aware Text-to-GQL.
Constructs structured prompts with physical schema context and partition grounding.
"""

from typing import List, Dict, Any, Optional
from partition_text2gql.core.types import SchemaElement, QueryIntent


class PromptBuilder:
    """Builds prompts for language model generation with physical partition annotations."""

    @staticmethod
    def build_generation_prompt(
        nl_question: str,
        domain: str,
        schema_elements: List[SchemaElement],
        intent: Optional[QueryIntent] = None,
        k_partitions: int = 4
    ) -> str:
        """
        Constructs the physical partition-grounded prompt.
        Format matches the training target used in Kaggle Notebook 3.
        """
        schema_lines = []
        for elem in schema_elements:
            elem_type = elem.element_type.name
            parts = f"[Partitions {elem.owning_partitions}]" if elem.owning_partitions else ""
            schema_lines.append(f"  - {elem_type}: {elem.name} {parts}")

        schema_context = "\n".join(schema_lines) if schema_lines else "  - Schema: General domain property graph"
        intent_hint = f" (Intent: {intent.value})" if intent else ""

        prompt = f"""You are an expert Graph Query Language (openCypher/GQL) engineer for a distributed graph database with {k_partitions} physical partitions.

Domain: {domain}
Physical Schema Context:
{schema_context}

User Question: {nl_question}{intent_hint}

Instructions:
1. Generate a syntactically valid openCypher read-only query.
2. If the traversal accesses a specific physical partition, annotate the MATCH clause with a `// partition:N` comment preceding the pattern.
3. Use exact labels and property names from the schema context.
4. Output ONLY the Cypher query.

Cypher Query:
"""
        return prompt.strip()

    @staticmethod
    def build_correction_prompt(
        nl_question: str,
        candidate_cypher: str,
        diagnostic_error: str,
        domain: str,
        schema_elements: List[SchemaElement]
    ) -> str:
        """Constructs a repair prompt for the self-correction feedback loop."""
        schema_lines = [f"  - {e.name} ({e.element_type.name})" for e in schema_elements]
        schema_str = "\n".join(schema_lines)

        prompt = f"""The following openCypher query generated for domain '{domain}' failed physical validation:

User Question: {nl_question}
Candidate Query:
{candidate_cypher}

Validation Diagnostic Error:
{diagnostic_error}

Valid Schema Elements for Domain:
{schema_str}

Please repair the query so it is syntactically valid, adheres to the domain schema, and respects read-only safety.
Output ONLY the corrected Cypher query:
"""
        return prompt.strip()
