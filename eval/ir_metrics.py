"""Information Retrieval (IR) Evaluation Metrics: Precision@K, Recall@K, MRR, and nDCG@K.

Provides standardized, mathematically rigorous ranking and retrieval metrics for compliance RAG.
"""

import math
import re
from typing import List, Dict, Any, Optional, Set, Union


def calculate_precision_at_k(
    retrieved_doc_ids: List[str],
    ground_truth_doc_ids: Union[List[str], Set[str]],
    k: int = 4
) -> float:
    """
    Calculate Precision@K: Ratio of relevant retrieved documents in top K.
    
    Formula:
        Precision@K = |{d in retrieved[:k] : d in ground_truth}| / k
    """
    if k <= 0:
        return 0.0
    if not retrieved_doc_ids or not ground_truth_doc_ids:
        return 0.0

    gt_set = set(ground_truth_doc_ids)
    top_k = list(dict.fromkeys(retrieved_doc_ids))[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in gt_set)
    return round(relevant_retrieved / k, 4)


def calculate_recall_at_k(
    retrieved_doc_ids: List[str],
    ground_truth_doc_ids: Union[List[str], Set[str]],
    k: int = 4
) -> float:
    """
    Calculate Recall@K: Proportion of relevant documents successfully retrieved in top K.
    
    Formula:
        Recall@K = |{d in retrieved[:k] : d in ground_truth}| / |ground_truth|
    """
    if k <= 0 or not ground_truth_doc_ids:
        return 0.0
    if not retrieved_doc_ids:
        return 0.0

    gt_set = set(ground_truth_doc_ids)
    if len(gt_set) == 0:
        return 0.0

    top_k = list(dict.fromkeys(retrieved_doc_ids))[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in gt_set)
    return round(relevant_retrieved / len(gt_set), 4)


def calculate_reciprocal_rank(
    retrieved_doc_ids: List[str],
    ground_truth_doc_ids: Union[List[str], Set[str]],
    k: Optional[int] = None
) -> float:
    """
    Calculate Reciprocal Rank (RR): Reciprocal of the rank of the first relevant document.
    
    Formula:
        RR = 1 / rank  (if first relevant document is at rank <= k; else 0.0)
    """
    if not retrieved_doc_ids or not ground_truth_doc_ids:
        return 0.0

    gt_set = set(ground_truth_doc_ids)
    ranked = list(dict.fromkeys(retrieved_doc_ids))
    candidates = ranked[:k] if k is not None and k > 0 else ranked

    for rank, doc_id in enumerate(candidates, start=1):
        if doc_id in gt_set:
            return round(1.0 / rank, 4)

    return 0.0


def calculate_dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at rank K.
    
    Formula:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)
    return dcg


def calculate_ndcg_at_k(
    retrieved_doc_ids: List[str],
    ground_truth_relevance: Union[List[str], Set[str], Dict[str, float]],
    k: int = 4
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at rank K (nDCG@K).
    Supports both binary ground truth (List/Set of doc_ids) and graded relevance (Dict[doc_id, score]).
    
    Formula:
        nDCG@K = DCG@K / IDCG@K
    """
    if k <= 0 or not retrieved_doc_ids or not ground_truth_relevance:
        return 0.0

    # Build relevance map
    if isinstance(ground_truth_relevance, dict):
        rel_map = ground_truth_relevance
        ideal_scores = sorted(rel_map.values(), reverse=True)
    else:
        gt_set = set(ground_truth_relevance)
        rel_map = {doc_id: 1.0 for doc_id in gt_set}
        ideal_scores = [1.0] * len(gt_set)

    # Actual retrieved relevance scores
    unique_retrieved = list(dict.fromkeys(retrieved_doc_ids))
    actual_scores = [rel_map.get(doc_id, 0.0) for doc_id in unique_retrieved[:k]]
    actual_dcg = calculate_dcg_at_k(actual_scores, k)

    # Ideal DCG
    ideal_dcg = calculate_dcg_at_k(ideal_scores, k)

    if ideal_dcg <= 0.0:
        return 0.0

    return round(min(1.0, actual_dcg / ideal_dcg), 4)


def evaluate_retrieval_ranking(
    retrieved_doc_ids: List[str],
    ground_truth_doc_ids: Union[List[str], Set[str]],
    k_values: List[int] = [1, 3, 4, 5, 10]
) -> Dict[str, float]:
    """
    Calculate a comprehensive suite of classic IR metrics for a single query.
    """
    metrics: Dict[str, float] = {}
    gt_set = set(ground_truth_doc_ids)

    metrics["mrr"] = calculate_reciprocal_rank(retrieved_doc_ids, gt_set)

    for k in k_values:
        metrics[f"precision@{k}"] = calculate_precision_at_k(retrieved_doc_ids, gt_set, k=k)
        metrics[f"recall@{k}"] = calculate_recall_at_k(retrieved_doc_ids, gt_set, k=k)
        metrics[f"ndcg@{k}"] = calculate_ndcg_at_k(retrieved_doc_ids, gt_set, k=k)

    return metrics


def calculate_ir_metrics_for_eval(
    retrieved_documents: List[Dict[str, Any]],
    expected_circular: Optional[str] = None,
    expected_clause: Optional[str] = None,
    expected_parent_ids: Optional[List[str]] = None,
    k: int = 4
) -> Dict[str, float]:
    """
    Evaluate retrieved documents against golden benchmark expectation.
    Matches either by explicit expected_parent_ids, or semantic matching on expected circular & clause.
    """
    retrieved_ids = list(dict.fromkeys(d.get("parent_id") or d.get("document_id") or f"doc_{idx}" for idx, d in enumerate(retrieved_documents)))
    
    # If explicit parent IDs given:
    if expected_parent_ids:
        gt_set = set(expected_parent_ids)
        return {
            f"precision@{k}": calculate_precision_at_k(retrieved_ids, gt_set, k=k),
            f"recall@{k}": calculate_recall_at_k(retrieved_ids, gt_set, k=k),
            "mrr": calculate_reciprocal_rank(retrieved_ids, gt_set, k=k),
            f"ndcg@{k}": calculate_ndcg_at_k(retrieved_ids, gt_set, k=k),
        }

    # Otherwise evaluate relevance dynamically based on expected circular / clause text
    relevance_scores: Dict[str, float] = {}
    circ_clean = (expected_circular or "").lower()
    clause_clean = (expected_clause or "").lower()
    # Extract clause numbers, e.g. "14", "8.2", "22"
    clause_nums = re.findall(r"\b\d+(?:\.\d+)?\b", clause_clean)

    for doc in retrieved_documents:
        doc_id = doc.get("parent_id") or doc.get("document_id") or ""
        doc_circ = (doc.get("circular_name") or "").lower()
        doc_clause = (doc.get("section_clause") or "").lower()
        doc_text = (doc.get("text") or "").lower()

        rel = 0.0
        # Check circular match
        if circ_clean and any(word in doc_circ for word in circ_clean.split() if len(word) > 4):
            rel += 0.5
        # Check clause number or keyword match
        if clause_nums and any(cn in doc_clause or cn in doc_text for cn in clause_nums):
            rel += 0.5
        elif clause_clean and any(cw in doc_clause for cw in clause_clean.split() if len(cw) > 4):
            rel += 0.5

        relevance_scores[doc_id] = min(1.0, rel)

    relevant_doc_ids = [did for did, score in relevance_scores.items() if score >= 0.5]
    
    return {
        f"precision@{k}": calculate_precision_at_k(retrieved_ids, relevant_doc_ids, k=k),
            f"recall@{k}": calculate_recall_at_k(retrieved_ids, relevant_doc_ids, k=k),
        "mrr": calculate_reciprocal_rank(retrieved_ids, relevant_doc_ids, k=k),
        f"ndcg@{k}": calculate_ndcg_at_k(retrieved_ids, relevance_scores, k=k),
    }
