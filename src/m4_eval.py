from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import math
import os, sys, json
import re
from dataclasses import asdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE))


def _overlap_score(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / math.sqrt(len(left_tokens) * len(right_tokens))


def _safe_float(value) -> float:
    try:
        score = float(value)
        if math.isnan(score):
            return 0.0
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def _fallback_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    per_question = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        context_text = "\n".join(ctxs or [])
        faithfulness = _overlap_score(answer, context_text)
        answer_relevancy = max(_overlap_score(question, answer), _overlap_score(ground_truth, answer))
        precision_values = [_overlap_score(question, ctx) for ctx in (ctxs or [])]
        context_precision = sum(precision_values) / len(precision_values) if precision_values else 0.0
        context_recall = _overlap_score(ground_truth, context_text)
        per_question.append(EvalResult(
            question=question,
            answer=answer,
            contexts=ctxs,
            ground_truth=ground_truth,
            faithfulness=_safe_float(faithfulness),
            answer_relevancy=_safe_float(answer_relevancy),
            context_precision=_safe_float(context_precision),
            context_recall=_safe_float(context_recall),
        ))

    aggregate = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        values = [getattr(item, metric) for item in per_question]
        aggregate[metric] = sum(values) / len(values) if values else 0.0
    aggregate["per_question"] = per_question
    return aggregate


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("questions, answers, contexts, and ground_truths must have the same length")

    if not LLM_API_KEY or os.getenv("LAB18_USE_REAL_RAGAS", "0") != "1":
        return _fallback_evaluation(questions, answers, contexts, ground_truths)

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        llm_kwargs = {"model": LLM_MODEL, "api_key": LLM_API_KEY, "temperature": 0}
        if LLM_BASE_URL:
            llm_kwargs["base_url"] = LLM_BASE_URL
        llm = ChatOpenAI(**llm_kwargs)
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_float(row.get("context_precision", 0.0)),
                context_recall=_safe_float(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]
        aggregate = {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            values = [getattr(item, metric) for item in per_question]
            aggregate[metric] = sum(values) / len(values) if values else _safe_float(result.get(metric, 0.0))
        aggregate["per_question"] = per_question
        return aggregate
    except Exception as e:
        print(f"  Warning: RAGAS evaluation failed, using fallback metrics ({e})", flush=True)
        return _fallback_evaluation(questions, answers, contexts, ground_truths)
    # 1. Wrap trong try/except — RAGAS cần OPENAI_API_KEY và Python 3.11+.
    # try:
    #     from ragas import evaluate
    #     from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    #     from datasets import Dataset
    #
    #     dataset = Dataset.from_dict({
    #         "question": questions, "answer": answers,
    #         "contexts": contexts, "ground_truth": ground_truths,
    #     })
    #     result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
    #                                         context_precision, context_recall])
    #     df = result.to_pandas()
    #     per_question = [EvalResult(question=row["question"], answer=row["answer"],
    #         contexts=row["contexts"], ground_truth=row["ground_truth"],
    #         faithfulness=float(row.get("faithfulness", 0.0)),
    #         answer_relevancy=float(row.get("answer_relevancy", 0.0)),
    #         context_precision=float(row.get("context_precision", 0.0)),
    #         context_recall=float(row.get("context_recall", 0.0)))
    #         for _, row in df.iterrows()]
    #     return {"faithfulness": ..., "answer_relevancy": ...,
    #             "context_precision": ..., "context_recall": ..., "per_question": [...]}
    # except Exception as e:
    #     print(f"  ⚠️  RAGAS evaluation failed: {e}")
    #     return zeros
    return {"faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating or unsupported by retrieved context", "Tighten the prompt and require citations from context."),
        "context_recall": ("Relevant evidence is missing from retrieved chunks", "Improve chunking, BM25 terms, or dense retrieval recall."),
        "context_precision": ("Retrieved context contains too much unrelated text", "Use reranking, metadata filters, or smaller child chunks."),
        "answer_relevancy": ("Answer does not directly address the question", "Refine the answer prompt and add stricter question matching."),
    }
    # 1. diagnostic_tree = {
    #        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
    #        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    #        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    #        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    #    }
    # 2. For each EvalResult: compute avg of 4 metrics, find worst_metric
    # 3. Sort by avg ascending → take bottom_n
    # 4. Return [{"question": ..., "worst_metric": ..., "score": ...,
    #             "diagnosis": ..., "suggested_fix": ...}]
    failures = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": result.question,
            "expected": result.ground_truth,
            "got": result.answer,
            "worst_metric": worst_metric,
            "score": _safe_float(metrics[worst_metric]),
            "avg_score": _safe_float(avg_score),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    failures.sort(key=lambda item: item["avg_score"])
    return failures[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "per_question": [asdict(item) for item in results.get("per_question", [])],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
