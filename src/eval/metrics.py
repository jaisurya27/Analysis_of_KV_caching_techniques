"""
Scoring functions used by LongBench.

Each task type uses a specific metric:
  - QA tasks (single-doc, multi-doc): F1 over bag-of-words tokens.
  - Summarization: ROUGE-L recall.
  - Code completion:  exact/edit-distance based score.
  - Few-shot:  classification accuracy.

Implementations follow the official LongBench evaluation code:
  https://github.com/THUDM/LongBench/blob/main/metrics.py
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    """Lower-case, remove punctuation/articles/extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _get_tokens(s: str) -> List[str]:
    return _normalize_answer(s).split()


# ---------------------------------------------------------------------------
# Token-level F1 (used for QA tasks)
# ---------------------------------------------------------------------------

def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between *prediction* and *ground_truth*."""
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(ground_truth)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score_zh(prediction: str, ground_truth: str) -> float:
    """Character-level F1 for Chinese QA tasks."""
    pred_chars = list(prediction)
    gold_chars = list(ground_truth)

    common = Counter(pred_chars) & Counter(gold_chars)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_chars)
    recall = num_same / len(gold_chars)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# ROUGE-L (used for summarization)
# ---------------------------------------------------------------------------

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of the Longest Common Subsequence of token lists."""
    m, n = len(x), len(y)
    # Space-optimised DP: O(min(m, n)) space.
    if m < n:
        x, y, m, n = y, x, n, m
    prev = [0] * (n + 1)
    for xi in x:
        curr = [0] * (n + 1)
        for j, yj in enumerate(y, 1):
            curr[j] = prev[j - 1] + 1 if xi == yj else max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 between *prediction* and *ground_truth*."""
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(ground_truth)

    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Classification accuracy (few-shot tasks)
# ---------------------------------------------------------------------------

def classification_score(prediction: str, ground_truth: str, all_classes: List[str]) -> float:
    """Return 1.0 if the predicted class matches, else 0.0.

    We check whether *ground_truth* appears in *prediction* and no other class
    label does (to avoid false positives from verbose outputs).
    """
    prediction_norm = prediction.strip().lower()
    ground_truth_norm = ground_truth.strip().lower()

    # Simple substring match, following the LongBench convention.
    if ground_truth_norm in prediction_norm:
        # Make sure no other class is also present.
        other_present = any(
            c.lower() != ground_truth_norm and c.lower() in prediction_norm
            for c in all_classes
        )
        if not other_present:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Code / edit-distance score (PassageCounting, Lcc, RepoBench-P)
# ---------------------------------------------------------------------------

def code_sim_score(prediction: str, ground_truth: str) -> float:
    """Normalised edit-distance similarity (1 - edit_dist / max_len)."""
    # Use difflib SequenceMatcher as a lightweight proxy.
    from difflib import SequenceMatcher

    pred = prediction.strip()
    gold = ground_truth.strip()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    return SequenceMatcher(None, pred, gold).ratio()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Map LongBench dataset names → scoring function key.
DATASET_TO_METRIC: dict[str, str] = {
    # Single-document QA
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1_zh",
    # Multi-document QA
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    # Chinese multi-document QA
    "dureader": "qa_f1_zh",
    # Summarization
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "vcsum": "rouge_l",
    # Few-shot classification
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l",
    "lsht": "classification",
    # Synthetic
    "passage_count": "classification",
    "passage_retrieval_en": "classification",
    "passage_retrieval_zh": "classification",
    # Code
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}

# Class labels for classification tasks.
CLASSIFICATION_LABELS: dict[str, list] = {
    "trec": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"],
    "lsht": [str(i) for i in range(1, 29)],
    "passage_count": [str(i) for i in range(1, 31)],
    "passage_retrieval_en": [str(i) for i in range(1, 31)],
    "passage_retrieval_zh": [str(i) for i in range(1, 31)],
}


def compute_metric(
    prediction: str,
    answers: List[str],
    dataset: str,
) -> float:
    """Compute the score for a single prediction against a list of gold answers.

    Returns the *maximum* score across all gold answers (standard practice).
    """
    metric = DATASET_TO_METRIC.get(dataset, "qa_f1")
    all_classes = CLASSIFICATION_LABELS.get(dataset, [])

    scores = []
    for ans in answers:
        if metric == "qa_f1":
            scores.append(qa_f1_score(prediction, ans))
        elif metric == "qa_f1_zh":
            scores.append(qa_f1_score_zh(prediction, ans))
        elif metric == "rouge_l":
            scores.append(rouge_l_score(prediction, ans))
        elif metric == "classification":
            scores.append(classification_score(prediction, ans, all_classes))
        elif metric == "code_sim":
            scores.append(code_sim_score(prediction, ans))
        else:
            scores.append(qa_f1_score(prediction, ans))

    return max(scores) if scores else 0.0
