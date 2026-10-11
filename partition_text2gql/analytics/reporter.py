"""
Analytics and Performance Comparison Reporter (Stage 5 & 6).
Generates investor-grade and publication-ready Markdown, LaTeX, and JSON comparison reports.
"""

from typing import List, Dict, Any, Optional
from partition_text2gql.core.types import CostComparisonReport


class AnalyticsReporter:
    """Formats side-by-side performance audits comparing Path A vs Path B."""

    @classmethod
    def to_markdown(cls, report: CostComparisonReport) -> str:
        """Generates clean, GitHub-flavored Markdown performance comparison report."""
        lines = [
            f"# Performance Audit: Path A (Naive) vs Path B (Partition-Aware)",
            f"",
            f"**Domain:** `{report.domain}`  ",
            f"**Natural Language Question:** *\"{report.nl_question}\"*  ",
            f"**Semantic Equivalence Verified:** `{'✅ YES (Identical Result Sets)' if report.semantic_equivalence else '❌ Discrepancy'}`",
            f"",
            f"---",
            f"",
            f"## 1. Executive Metric Summary",
            f"",
            f"| Metric | Path A (Naive Scatter-Gather) | Path B (Cost-Based Partition-Aware) | Improvement / Reduction |",
            f"| :--- | :--- | :--- | :--- |",
            f"| **Execution Latency** | `{report.path_a_metrics.latency_ms:.2f} ms` | `{report.path_b_metrics.latency_ms:.2f} ms` | **{report.latency_speedup_factor:.1f}x Speedup** |",
            f"| **Cross-Partition Network Hops** | `{report.path_a_metrics.cross_partition_hops}` | `{report.path_b_metrics.cross_partition_hops}` | **{report.hop_reduction_pct:.1f}% Reduction** |",
            f"| **Network Shuffle Volume** | `{report.path_a_metrics.network_shuffle_volume_bytes:,} bytes` | `{report.path_b_metrics.network_shuffle_volume_bytes:,} bytes` | **{report.shuffle_volume_reduction_pct:.1f}% Reduction** |",
            f"| **Subqueries Dispatched** | `{report.path_a_metrics.subqueries_dispatched}` | `{report.path_b_metrics.subqueries_dispatched}` | Localized Subtree Pushdown |",
            f"| **Intermediate Rows Shuffled** | `{report.path_a_metrics.intermediate_rows_shuffled}` | `{report.path_b_metrics.intermediate_rows_shuffled}` | Filter Pushdown Pruning |",
            f"",
            f"---",
            f"",
            f"## 2. Distributed Architecture Insight",
            f"- **Path A (Naive)** broadcast the unpartitioned Cypher pattern to all shards. Shards evaluated full cartesian scans without predicate locality, incurring high RPC round-trips.",
            f"- **Path B (Patent Innovation)** decomposed the pattern at physical partition cut boundaries, pushed filter predicates directly to owning shards, and synthesized minimal boundary joins on join keys only.",
            f""
        ]
        return "\n".join(lines)

    @classmethod
    def to_latex(cls, reports: List[CostComparisonReport]) -> str:
        """Exports publication-grade LaTeX booktabs table for research papers / patent applications."""
        latex = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Ablation and Performance Comparison: Naive Baseline (Path A) vs. Partition-Aware Engine (Path B).}",
            r"\label{tab:path_a_vs_path_b}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"\textbf{Domain} & \multicolumn{2}{c}{\textbf{Latency (ms)}} & \multicolumn{2}{c}{\textbf{Network Hops}} & \multicolumn{2}{c}{\textbf{Shuffle Volume (Bytes)}} \\",
            r" & Path A & Path B & Path A & Path B & Path A & Path B \\",
            r"\midrule"
        ]

        for r in reports:
            latex.append(
                f"{r.domain.capitalize()} & "
                f"{r.path_a_metrics.latency_ms:.1f} & \\textbf{{{r.path_b_metrics.latency_ms:.1f}}} & "
                f"{r.path_a_metrics.cross_partition_hops} & \\textbf{{{r.path_b_metrics.cross_partition_hops}}} & "
                f"{r.path_a_metrics.network_shuffle_volume_bytes:,} & \\textbf{{{r.path_b_metrics.network_shuffle_volume_bytes:,}}} \\\\"
            )

        latex.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}"
        ])
        return "\n".join(latex)

