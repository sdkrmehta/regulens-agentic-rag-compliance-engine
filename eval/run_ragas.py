"""Throttled Ragas Evaluation Runner comparing Naive Baseline vs V3.0 Agentic Hybrid RAG."""

import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from src.retrieval.vector_store import RBIHybridVectorStore
from src.agent.graph import compliance_agent
from eval.baseline_rag import NaiveBaselineRAG


def calculate_token_overlap_metrics(
    retrieved_contexts: List[str],
    ground_truth: str,
    answer: str
) -> Dict[str, float]:
    """
    Deterministic evaluation metrics approximating Context Precision, Context Recall,
    and Faithfulness based on statutory term overlap.
    """
    gt_words = set(ground_truth.lower().split()) - {"the", "a", "an", "is", "of", "and", "in", "to", "for", "with", "shall", "under"}
    if not gt_words:
        gt_words = set(ground_truth.lower().split())

    all_context = " ".join(retrieved_contexts).lower()
    ans_words = set(answer.lower().split())

    # Context Recall: proportion of ground-truth words captured in retrieved context
    recalled = sum(1 for w in gt_words if w in all_context)
    context_recall = recalled / max(len(gt_words), 1)

    # Context Precision: proportion of retrieved context sentences that contain ground truth terms
    if retrieved_contexts:
        relevant_chunks = sum(1 for c in retrieved_contexts if any(w in c.lower() for w in list(gt_words)[:5]))
        context_precision = relevant_chunks / len(retrieved_contexts)
    else:
        context_precision = 0.0

    # Faithfulness: proportion of answer claims supported by retrieved context
    if ans_words:
        grounded = sum(1 for w in ans_words if w in all_context)
        faithfulness = grounded / max(len(ans_words), 1)
    else:
        faithfulness = 0.0

    # Answer Relevancy
    relevancy = sum(1 for w in gt_words if w in ans_words) / max(len(gt_words), 1)

    return {
        "context_precision": min(1.0, round(context_precision, 4)),
        "context_recall": min(1.0, round(context_recall, 4)),
        "faithfulness": min(1.0, round(faithfulness, 4)),
        "answer_relevancy": min(1.0, round(relevancy, 4)),
    }


def run_benchmark(
    golden_dataset_path: str = "data/golden_eval_25.json",
    output_csv_path: str = "eval/benchmark_results.csv",
    sample_size: int = 25,
    cooldown_seconds: float = 0.5,
):
    print(f"Loading golden dataset from {golden_dataset_path}...")
    with open(golden_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)[:sample_size]

    # Initialize systems
    print("Initializing Naive Baseline RAG...")
    baseline = NaiveBaselineRAG()
    baseline.build_baseline_index()

    vector_store = RBIHybridVectorStore()

    records = []
    baseline_totals = {"precision": 0.0, "recall": 0.0, "faithfulness": 0.0, "relevancy": 0.0}
    agentic_totals = {"precision": 0.0, "recall": 0.0, "faithfulness": 0.0, "relevancy": 0.0}

    print(f"\nRunning sequential benchmark across {len(dataset)} questions with {cooldown_seconds}s cooldown...")

    for i, item in enumerate(dataset, 1):
        q_id = item["question_id"]
        q_text = item["question"]
        gt = item["ground_truth"]
        tier = item.get("tier", "General")

        print(f"[{i}/{len(dataset)}] Evaluating {q_id} ({tier}): {q_text[:60]}...")

        # 1. Run Baseline
        b_res = baseline.query(q_text, top_k=4)
        b_metrics = calculate_token_overlap_metrics(b_res["retrieved_contexts"], gt, b_res["answer"])

        # 2. Run Agentic Hybrid V3.0
        # Retrieve through hybrid store
        hybrid_parents = vector_store.hybrid_search_and_rerank(query=q_text, top_k=4)
        hybrid_contexts = [p["text"] for p in hybrid_parents]
        
        # Invoke agent for synthesis
        agent_out = compliance_agent.invoke({"question": q_text, "retry_count": 0})
        agent_resp = agent_out.get("final_response")
        agent_answer = agent_resp.answer if agent_resp else "No response"

        a_metrics = calculate_token_overlap_metrics(hybrid_contexts, gt, agent_answer)

        # Accumulate
        for k, v in [("precision", "context_precision"), ("recall", "context_recall"), ("faithfulness", "faithfulness"), ("relevancy", "answer_relevancy")]:
            baseline_totals[k] += b_metrics[v]
            agentic_totals[k] += a_metrics[v]

        records.append({
            "question_id": q_id,
            "tier": tier,
            "question": q_text,
            "baseline_precision": b_metrics["context_precision"],
            "agentic_precision": a_metrics["context_precision"],
            "baseline_recall": b_metrics["context_recall"],
            "agentic_recall": a_metrics["context_recall"],
            "baseline_faithfulness": b_metrics["faithfulness"],
            "agentic_faithfulness": a_metrics["faithfulness"],
        })

        time.sleep(cooldown_seconds)

    # Compute averages
    n = len(dataset)
    b_avg_p = round(baseline_totals["precision"] / n, 4)
    b_avg_r = round(baseline_totals["recall"] / n, 4)
    b_avg_f = round(baseline_totals["faithfulness"] / n, 4)

    a_avg_p = round(agentic_totals["precision"] / n, 4)
    a_avg_r = round(agentic_totals["recall"] / n, 4)
    a_avg_f = round(agentic_totals["faithfulness"] / n, 4)

    print("\n" + "=" * 65)
    print("BENCHMARK ABLATION RESULTS (Naive Baseline vs Agentic V3.0)")
    print("=" * 65)
    print(f"Metric             | Naive Baseline | Agentic Hybrid V3.0 | Delta")
    print("-" * 65)
    print(f"Context Precision  | {b_avg_p:<14} | {a_avg_p:<19} | +{(a_avg_p - b_avg_p)*100:.1f}%")
    print(f"Context Recall     | {b_avg_r:<14} | {a_avg_r:<19} | +{(a_avg_r - b_avg_r)*100:.1f}%")
    print(f"Faithfulness       | {b_avg_f:<14} | {a_avg_f:<19} | +{(a_avg_f - b_avg_f)*100:.1f}%")
    print("=" * 65)

    # Save CSV
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "tier", "question",
            "baseline_precision", "agentic_precision",
            "baseline_recall", "agentic_recall",
            "baseline_faithfulness", "agentic_faithfulness"
        ])
        writer.writeheader()
        writer.writerows(records)

    print(f"[SUCCESS] Exported detailed ablation metrics to {out_path}")
    return {
        "baseline_precision": b_avg_p,
        "agentic_precision": a_avg_p,
        "baseline_recall": b_avg_r,
        "agentic_recall": a_avg_r,
        "baseline_faithfulness": b_avg_f,
        "agentic_faithfulness": a_avg_f,
    }


if __name__ == "__main__":
    run_benchmark()
