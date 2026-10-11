"""Configuration management for the Partition-Aware Text2GQL system."""

from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Runtime configuration knobs for planning, execution, and instrumentation."""
    # Partition settings
    default_k: int = 4
    seed: int = 42

    # Execution sandbox
    shard_timeout_seconds: float = 5.0
    max_worker_threads: int = 8
    max_result_rows: int = 10000

    # Cost model hyperparameters (calibrated from distributed benchmarks)
    cost_weight_hop: float = Field(default=50.0, description="Latency penalty (ms) per cross-partition hop")
    cost_weight_byte_shuffle: float = Field(default=0.005, description="Cost penalty per network shuffle byte")
    cost_weight_local_scan: float = Field(default=0.001, description="Local vertex/edge scan cost per item")

    # Safety limits
    max_path_hops_allowed: int = 6
    disallow_unbounded_paths: bool = True
    disallow_write_operations: bool = True

