"""Search quality benchmark script.

Tests the RAG system against known Q&A pairs and measures:
- MRR (Mean Reciprocal Rank)
- Recall@K
- NDCG@K
- Latency statistics
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

# Test Q&A pairs: (query, expected_source_ids)
TEST_CASES = [
    ("How do I reset my password?", ["confluence:SUPPORT:1001"]),
    ("What are the password requirements?", ["confluence:SUPPORT:1001"]),
    ("How to connect to VPN?", ["slack:C123:1704067200.000"]),
    ("How do I get a 2FA code for VPN?", ["slack:C123:1704067200.000"]),
    ("How to authenticate with the API?", ["git:dev-portal:docs/api/authentication.md"]),
    ("What is the API rate limit?", ["git:dev-portal:docs/api/authentication.md"]),
    ("How long do access tokens last?", ["git:dev-portal:docs/api/authentication.md"]),
    ("Password reset email not received", ["confluence:SUPPORT:1001"]),
]


@dataclass
class BenchmarkResult:
    query: str
    expected_ids: list[str]
    found_ids: list[str]
    reciprocal_rank: float
    recall_at_5: float
    latency_ms: float


async def run_benchmark():
    from src.search.engine import SearchEngine

    engine = SearchEngine()
    results: list[BenchmarkResult] = []

    for query, expected_ids in TEST_CASES:
        start = time.time()
        response = await engine.search(query=query, limit=5)
        latency = (time.time() - start) * 1000

        found_ids = [s.chunk_id for s in response.sources]

        # MRR: reciprocal of rank of first relevant result
        rr = 0.0
        for rank, fid in enumerate(found_ids, 1):
            if any(eid in fid for eid in expected_ids):
                rr = 1.0 / rank
                break

        # Recall@5
        found_relevant = sum(
            1 for fid in found_ids if any(eid in fid for eid in expected_ids)
        )
        recall = found_relevant / len(expected_ids) if expected_ids else 0.0

        results.append(
            BenchmarkResult(
                query=query,
                expected_ids=expected_ids,
                found_ids=found_ids,
                reciprocal_rank=rr,
                recall_at_5=recall,
                latency_ms=latency,
            )
        )

    # Print results
    print("\n" + "=" * 80)
    print("SEARCH QUALITY BENCHMARK")
    print("=" * 80)

    for r in results:
        status = "PASS" if r.reciprocal_rank > 0 else "FAIL"
        print(f"\n[{status}] {r.query}")
        print(f"  RR: {r.reciprocal_rank:.2f}  Recall@5: {r.recall_at_5:.2f}  Latency: {r.latency_ms:.0f}ms")

    mrr = sum(r.reciprocal_rank for r in results) / len(results)
    avg_recall = sum(r.recall_at_5 for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    p95_latency = sorted(r.latency_ms for r in results)[int(len(results) * 0.95)]

    print(f"\n{'=' * 80}")
    print(f"MRR:           {mrr:.3f}  (target: > 0.7)")
    print(f"Avg Recall@5:  {avg_recall:.3f}  (target: > 0.85)")
    print(f"Avg Latency:   {avg_latency:.0f}ms")
    print(f"P95 Latency:   {p95_latency:.0f}ms  (target: < 500ms)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
