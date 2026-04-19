"""Prometheus metrics. Scraped by the existing observability stack — see
prometheus scrape config in vps-infrastructure/roles/monitoring/."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Labels are bounded on purpose — no per-merchant / per-tx cardinality.
predictions_total = Counter(
    "categorizer_predictions_total",
    "Total classification calls, labeled by the tier that emitted the answer.",
    ["tier", "status"],  # tier ∈ {rules, knn, llm_notink, llm_think, reject}
)

corrections_total = Counter(
    "categorizer_corrections_total",
    "User corrections, labeled by whether the parent was already right.",
    ["parent_was_correct"],  # "true" | "false"
)

categorize_latency_seconds = Histogram(
    "categorizer_latency_seconds",
    "Wall-clock latency per /v1/categorize call.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0),
)

tier_latency_seconds = Histogram(
    "categorizer_tier_latency_seconds",
    "Latency per tier (broken out).",
    ["tier"],
    buckets=(0.005, 0.025, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 45.0),
)

embedding_time_seconds = Histogram(
    "categorizer_embedding_seconds",
    "Time to compute one embedding (fastembed).",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
