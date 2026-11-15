"""Compatibility entry point for the retrieval evaluation report."""

from rag.evaluation import run_answer_evaluation, run_evaluation


if __name__ == "__main__":
    from rag.retriever import retrieve

    report = run_evaluation(retrieve)
    print(f"Evidence checks passed: {report['passed']}/{report['total']}")
    print(f"Needs review: {report['needs_review']}")
    print(f"Precision@{report['k']}: {report['precision_percentage']}%")
    print(f"Recall@{report['k']}: {report['recall_percentage']}%")
    print(f"F1@{report['k']}: {report['f1_percentage']}%")
    print(f"Quality: {report['quality_label']}")
