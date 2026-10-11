"""
Query Validator & Self-Correction Feedback Loop (Stage 3 -> Stage 2).
Automates iterative query repair when syntax, schema, or safety validations fail.
"""

from typing import Optional, Callable, Dict, Any, Tuple, List
from pydantic import BaseModel, Field
from partition_text2gql.parser.ast_nodes import QueryAST
from partition_text2gql.validator.validators import (
    SyntaxValidator, SchemaConformanceChecker, AccessSafetyChecker
)
from partition_text2gql.validator.schema_catalog import GraphDomainSchema
from partition_text2gql.core.errors import (
    Text2GQLError, CypherSyntaxError, SchemaValidationError, SecurityViolationError
)


class CorrectionAttempt(BaseModel):
    attempt_index: int
    candidate_query: str
    error_type: str
    error_message: str
    corrective_prompt: str


class CorrectionResult(BaseModel):
    is_valid: bool
    final_query: str
    final_ast: Optional[Any] = None
    retries_used: int = 0
    attempts: List[CorrectionAttempt] = Field(default_factory=list)


class SelfCorrectionFeedbackLoop:
    """
    Closed-Loop Query Validation & Correction:
    Intercepts syntax errors, schema hallucinations, and security violations.
    Constructs diagnostic correction prompts and orchestrates iterative LLM re-prompting.
    """

    def __init__(
        self,
        schema: GraphDomainSchema,
        max_retries: int = 2,
        safety_hops: int = 6
    ):
        self.schema = schema
        self.max_retries = max_retries
        self.syntax_validator = SyntaxValidator()
        self.schema_checker = SchemaConformanceChecker(schema)
        self.safety_checker = AccessSafetyChecker(max_hops=safety_hops)

    def validate_and_repair(
        self,
        nl_question: str,
        initial_query: str,
        generator_fn: Optional[Callable[[str], str]] = None
    ) -> CorrectionResult:
        """
        Validates query. If validation fails and generator_fn is provided,
        iteratively feeds diagnostic errors back to the generator up to max_retries.
        """
        current_query = initial_query
        attempts: List[CorrectionAttempt] = []

        for attempt_idx in range(self.max_retries + 1):
            try:
                # 1. Safety text check
                self.safety_checker.validate_text(current_query)

                # 2. Formal AST syntax check
                ast = self.syntax_validator.validate(current_query)

                # 3. Schema conformance check
                self.schema_checker.validate(ast)

                # 4. AST safety check
                self.safety_checker.validate_ast(ast)

                # Success!
                return CorrectionResult(
                    is_valid=True,
                    final_query=current_query,
                    final_ast=ast,
                    retries_used=attempt_idx,
                    attempts=attempts
                )

            except (CypherSyntaxError, SchemaValidationError, SecurityViolationError) as err:
                error_type = type(err).__name__
                error_msg = str(err)

                # Construct structured corrective feedback prompt
                corrective_prompt = self._construct_correction_prompt(
                    nl_question=nl_question,
                    failed_query=current_query,
                    error_type=error_type,
                    error_msg=error_msg
                )

                attempts.append(CorrectionAttempt(
                    attempt_index=attempt_idx,
                    candidate_query=current_query,
                    error_type=error_type,
                    error_message=error_msg,
                    corrective_prompt=corrective_prompt
                ))

                if attempt_idx < self.max_retries and generator_fn is not None:
                    # Invoke generator with corrective prompt
                    current_query = generator_fn(corrective_prompt)
                else:
                    # Exceeded retries or no generator available
                    break

        return CorrectionResult(
            is_valid=False,
            final_query=current_query,
            final_ast=None,
            retries_used=len(attempts),
            attempts=attempts
        )

    def _construct_correction_prompt(
        self,
        nl_question: str,
        failed_query: str,
        error_type: str,
        error_msg: str
    ) -> str:
        """Builds a zero-shot reflection prompt highlighting the exact failure mode."""
        allowed_labels = list(self.schema.labels.keys())[:10]
        allowed_rels = list(self.schema.relationships.keys())[:10]

        prompt = (
            f"### Query Validation Feedback Loop\n"
            f"The generated openCypher query failed validation with error: {error_type}\n"
            f"Diagnostic Message: {error_msg}\n\n"
            f"Original Question: \"{nl_question}\"\n"
            f"Failed Query:\n```cypher\n{failed_query}\n```\n\n"
            f"Available Schema Labels: {allowed_labels}\n"
            f"Available Relationship Types: {allowed_rels}\n\n"
            f"Instruction: Correct the query to adhere strictly to the openCypher grammar and domain schema. "
            f"Do not invent non-existent labels. Output only the corrected Cypher query."
        )
        return prompt

