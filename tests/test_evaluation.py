import tempfile
import unittest
from pathlib import Path

from rag.evaluation import (
    context_precision,
    load_evaluation_history,
    precision_at_k,
    r_precision_at_k,
    record_evaluation_run,
    run_answer_evaluation,
    summarize_failures,
    _term_matches,
)


class EvaluationTests(unittest.TestCase):
    def test_precision_variants(self):
        ranked = ["noise", "relevant", "extra"]
        relevant = ["relevant"]
        self.assertEqual(precision_at_k(ranked, relevant, 5), 0.2)
        self.assertEqual(r_precision_at_k(ranked, relevant, 5), 0.0)
        self.assertEqual(context_precision(ranked[:2], relevant), 0.5)

    def test_full_answer_cache_and_tolerant_percentage_match(self):
        calls = []

        def retrieve(question, k):
            return [{"id": "chunk-1", "text": "evidence", "source": "doc.txt"}]

        def build_context(chunks):
            return "stable context", []

        def generate(question, context):
            calls.append((question, context))
            return "The prerequisites for AI-101 include Basic Python and introductory statistics [S1]."

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "answers.json"
            first = run_answer_evaluation(
                retrieve,
                build_context,
                generate,
                limit=1,
                cache_path=cache_path,
                model="test-model",
                delay=0,
            )
            second = run_answer_evaluation(
                retrieve,
                build_context,
                generate,
                limit=1,
                cache_path=cache_path,
                model="test-model",
                delay=0,
            )

        self.assertEqual(first["api_calls"], 1)
        self.assertEqual(second["cache_hits"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["results"][0]["term_completeness"], 1)
        self.assertTrue(_term_matches("25 percent", "The result is 25%."))

    def test_history_round_trip_and_failure_summary(self):
        report = {
            "k": 5,
            "total": 1,
            "passed": 0,
            "needs_review": 1,
            "precision_percentage": 20.0,
            "capped_precision_percentage": 100.0,
            "context_precision_percentage": 100.0,
            "recall_percentage": 100.0,
            "f1_percentage": 33.33,
            "mrr_percentage": 100.0,
            "no_answer_percentage": 100.0,
            "results": [{
                "question": "Failed question",
                "expected_source": "doc.txt",
                "passed": False,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.jsonl"
            record_evaluation_run(report, history_path=history_path)
            runs = load_evaluation_history(history_path)

        failures = summarize_failures(runs)
        self.assertEqual(len(runs), 1)
        self.assertEqual(failures[0]["question"], "Failed question")
        self.assertEqual(failures[0]["source"], "doc.txt")


if __name__ == "__main__":
    unittest.main()
