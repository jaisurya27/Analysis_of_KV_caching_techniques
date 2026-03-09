"""
LongBench evaluation pipeline.

LongBench (THUDM/LongBench on HuggingFace Datasets) provides 21 English and
Chinese long-context tasks covering QA, summarization, few-shot learning, code
completion, and synthetic retrieval tasks.

Reference:
  Bai et al., "LongBench: A Bilingual, Multitask Benchmark for Long Context
  Understanding", ACL 2024.  https://arxiv.org/abs/2308.14508
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.eval.metrics import DATASET_TO_METRIC, compute_metric
from src.models.patch import create_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LongBench data download helper
# ---------------------------------------------------------------------------

_LONGBENCH_ZIP_URL = (
    "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
)
_LONGBENCH_CACHE_DIR = Path.home() / ".cache" / "longbench_data"


def _ensure_longbench_data(cache_dir: Path = _LONGBENCH_CACHE_DIR) -> Path:
    """Download and extract LongBench data.zip on first use.

    The THUDM/LongBench HF repo no longer supports script-based loading
    (datasets >= 3.0) and stores all task JSONL files inside a single
    data.zip archive.  This helper downloads the zip once (~114 MB) and
    extracts it to *cache_dir*, then returns the directory containing
    the *.jsonl files.
    """
    # Check if already extracted (any .jsonl present at root or in data/)
    for search_dir in [cache_dir, cache_dir / "data"]:
        if search_dir.exists() and any(search_dir.glob("*.jsonl")):
            logger.info("LongBench data already cached at %s", search_dir)
            return search_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "data.zip"

    logger.info("Downloading LongBench data.zip (~114 MB) …")
    print("Downloading LongBench data.zip (~114 MB) — one-time download …")
    urllib.request.urlretrieve(_LONGBENCH_ZIP_URL, zip_path)

    logger.info("Extracting data.zip to %s …", cache_dir)
    print(f"Extracting to {cache_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)
    zip_path.unlink()  # remove zip after extraction

    # Return whichever directory actually contains the jsonl files
    for search_dir in [cache_dir, cache_dir / "data"]:
        if search_dir.exists() and any(search_dir.glob("*.jsonl")):
            return search_dir

    raise FileNotFoundError(
        f"Extraction succeeded but no *.jsonl files found under {cache_dir}. "
        "Check the zip structure manually."
    )

# ---------------------------------------------------------------------------
# Task prompt templates (same as original LongBench)
# ---------------------------------------------------------------------------

DATASET_PROMPTS: Dict[str, str] = {
    "narrativeqa": (
        "You are given a story and a question. Answer the question based on the story.\n\n"
        "Story: {context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific paper and a question. Answer the question based on the paper.\n\n"
        "Paper: {context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer the question.\n\n"
        "Text: {context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Passages: {context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Passages: {context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Passages: {context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report: {context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query. Summarize the relevant part of the meeting.\n\n"
        "Transcript: {context}\n\nQuery: {input}\n\nSummary:"
    ),
    "multi_news": (
        "You are given several news pieces. Write a one-page summary.\n\n"
        "{context}\n\nNow, write a one-page summary of all the news. Summary:"
    ),
    "trec": (
        "{context}\n\n{input}"
    ),
    "triviaqa": (
        "{context}\n\nAnswer the question based on the given passage. "
        "Only give me the answer and do not output any other words. "
        "The following are some examples.\n\n{input}"
    ),
    "samsum": (
        "{context}\n\nSummarize the above dialogue.\n\nSummary:"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be repetitions. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\nThe output format should be a single integer, for example: 5. "
        "Answer:"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\nThe following is an abstract.\n\n{input}\n\n"
        "The answer should be a single integer paragraph number, for example: 4. "
        "Answer:"
    ),
    "lcc": (
        "Please complete the code given below.\n{context}Output:"
    ),
    "repobench-p": (
        "{context}Output:"
    ),
}

# Tasks we run by default (a representative, tractable subset).
DEFAULT_TASKS: List[str] = [
    "narrativeqa",
    "qasper",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "trec",
    "passage_count",
]

# Max generation tokens per task type.
MAX_NEW_TOKENS: Dict[str, int] = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(example: dict, dataset: str, tokenizer, max_length: int = 3800) -> str:
    """Construct the input prompt for a LongBench example.

    Long contexts are truncated from the middle (keeping start and end) to
    stay within *max_length* tokens.
    """
    template = DATASET_PROMPTS.get(
        dataset,
        "Context: {context}\n\nQuestion: {input}\n\nAnswer:",
    )

    context = example.get("context", "")
    question = example.get("input", "")

    # Truncate context if necessary.
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    # Reserve tokens for the rest of the prompt.
    overhead = len(tokenizer.encode(
        template.format(context="", input=question),
        add_special_tokens=False,
    ))
    budget = max_length - overhead - 50  # 50-token safety margin

    if len(context_tokens) > budget and budget > 0:
        half = budget // 2
        context_tokens = context_tokens[:half] + context_tokens[-half:]
        context = tokenizer.decode(context_tokens, skip_special_tokens=True)

    return template.format(context=context, input=question)


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class LongBenchEvaluator:
    """Run LongBench evaluation for a given model and KV-compression method.

    Args:
        model:          Patched (or unpatched) causal-LM.
        tokenizer:      Matching tokenizer.
        method:         KV method name ("full", "h2o", "streaming_llm").
        tasks:          List of LongBench task names to evaluate.
        max_length:     Maximum context length fed to the model (tokens).
        num_samples:    Limit samples per task (None = all).
        output_dir:     Directory to write per-task result JSON files.
        device:         Device string, e.g. "cuda", "mps", "cpu".
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        method: str,
        tasks: List[str] = DEFAULT_TASKS,
        max_length: int = 3800,
        num_samples: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        output_dir: str = "results",
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.tasks = tasks
        self.max_length = max_length
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens  # overrides per-task defaults if set
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        """Run model.generate for a single prompt and return the output text."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]

        cache = create_cache(self.model, self.method)
        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )
        if cache is not None:
            gen_kwargs["past_key_values"] = cache

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0, input_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Task evaluation
    # ------------------------------------------------------------------

    def evaluate_task(self, dataset: str) -> Dict:
        """Evaluate a single LongBench task; return a result dict."""
        logger.info("Loading LongBench task: %s", dataset)

        data_dir = _ensure_longbench_data()
        jsonl_path = data_dir / f"{dataset}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Task file not found: {jsonl_path}\n"
                f"Available files: {sorted(data_dir.glob('*.jsonl'))}"
            )
        raw = load_dataset("json", data_files=str(jsonl_path), split="train")
        if self.num_samples is not None:
            raw = raw.select(range(min(self.num_samples, len(raw))))

        max_new = self.max_new_tokens if self.max_new_tokens is not None else MAX_NEW_TOKENS.get(dataset, 128)
        scores, predictions, timings = [], [], []

        for example in tqdm(raw, desc=f"{dataset} [{self.method}]"):
            prompt = build_prompt(
                example, dataset, self.tokenizer, self.max_length
            )
            t0 = time.perf_counter()
            pred = self._generate(prompt, max_new)
            elapsed = time.perf_counter() - t0

            answers = example.get("answers", [example.get("answer", "")])
            if isinstance(answers, str):
                answers = [answers]

            score = compute_metric(pred, answers, dataset)
            scores.append(score)
            predictions.append(pred)
            timings.append(elapsed)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_time = sum(timings) / len(timings) if timings else 0.0

        result = {
            "dataset": dataset,
            "method": self.method,
            "num_samples": len(scores),
            "metric": DATASET_TO_METRIC.get(dataset, "qa_f1"),
            "score": round(avg_score * 100, 2),
            "avg_latency_sec": round(avg_time, 3),
            "predictions": predictions,
            "individual_scores": [round(s * 100, 2) for s in scores],
        }

        out_path = self.output_dir / f"{dataset}_{self.method}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(
            "  %-30s  score=%.2f%%  avg_latency=%.3fs",
            dataset,
            result["score"],
            avg_time,
        )
        return result

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Dict]:
        """Evaluate all requested tasks; return a dict of results keyed by task."""
        all_results: Dict[str, Dict] = {}
        for task in self.tasks:
            try:
                all_results[task] = self.evaluate_task(task)
            except Exception as exc:
                logger.error("Task %s failed: %s", task, exc, exc_info=True)

        # Write summary.
        summary = {
            "method": self.method,
            "tasks": {
                t: {"score": r["score"], "avg_latency_sec": r["avg_latency_sec"]}
                for t, r in all_results.items()
            },
            "macro_avg_score": round(
                sum(r["score"] for r in all_results.values()) / max(1, len(all_results)),
                2,
            ),
        }
        summary_path = self.output_dir / f"summary_{self.method}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "\n=== %s — Macro-avg score: %.2f%% ===",
            self.method.upper(),
            summary["macro_avg_score"],
        )
        return all_results
