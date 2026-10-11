"""
FastAPI Microservice for Partition-Aware Text-to-GQL.
Exposes REST endpoints for querying, performance auditing, and shard health checking.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time

from partition_text2gql.pipeline import PartitionAwareText2GQLEngine
from partition_text2gql.validator.schema_catalog import GraphDomainSchema
from partition_text2gql.stats.store import PartitionStatisticsStore
from partition_text2gql.executor.shard import GraphShard
from partition_text2gql.core.types import CostComparisonReport

app = FastAPI(
    title="Partition-Aware Text-to-GQL Engine API",
    description="Enterprise Distributed Graph Translation Service with Cost-Based Partition Optimization",
    version="1.0.0"
)

# In-memory engine registry per domain
ENGINES: Dict[str, PartitionAwareText2GQLEngine] = {}


class QueryRequest(BaseModel):
    nl_question: str = Field(description="Natural language question")
    domain: str = Field(default="movie", description="Graph domain name (movie, nba, company)")
    gold_or_candidate_cypher: Optional[str] = Field(default=None, description="Optional raw candidate Cypher")


class QueryResponse(BaseModel):
    domain: str
    nl_question: str
    results: List[Dict[str, Any]]
    comparison_audit: Dict[str, Any]
    execution_time_ms: float


def init_default_engines():
    """Initializes domain engines and partition stores on service boot."""
    stats_store = PartitionStatisticsStore()

    # Pre-seed Movie Domain Engine
    movie_schema = GraphDomainSchema(domain="movie")
    movie_schema.add_label("Movie", ["title", "release_year"])
    movie_schema.add_label("Person", ["name"])
    movie_schema.add_relationship("directedBy", source="Movie", target="Person")

    shards = {0: GraphShard(0), 1: GraphShard(1), 2: GraphShard(2), 3: GraphShard(3)}
    stats_store.register_domain_partitioning("movie", {}, 4)

    ENGINES["movie"] = PartitionAwareText2GQLEngine("movie", movie_schema, stats_store, shards)

# Initialize on boot
init_default_engines()


@app.get("/health")
def health_check():
    """Verifies service liveness and active shard engines."""
    return {
        "status": "HEALTHY",
        "timestamp": time.time(),
        "active_domains": list(ENGINES.keys()),
        "shards_per_domain": 4
    }


@app.post("/query", response_model=QueryResponse)
def execute_query(req: QueryRequest):
    """Executes natural language query through the 6-stage partition-aware pipeline."""
    engine = ENGINES.get(req.domain)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Domain '{req.domain}' is not registered.")

    cypher = req.gold_or_candidate_cypher or "MATCH (m:Movie) RETURN m.title LIMIT 10"
    t0 = time.perf_counter()

    try:
        results, report = engine.process_natural_language(req.nl_question, cypher)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        return QueryResponse(
            domain=req.domain,
            nl_question=req.nl_question,
            results=results,
            comparison_audit=report.model_dump(),
            execution_time_ms=round(duration_ms, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
