"""Compatibility entry point for the retrieval evaluation report."""

import argparse

from rag.evaluation import (
    _select_history_run,
    compare_evaluation_runs,
    load_evaluation_history,
    record_evaluation_run,
    run_answer_evaluation,
    run_evaluation,
    summarize_failures,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run retrieval or full answer evaluation.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the opt-in answer evaluation, which may call the model API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit full evaluation to the first N cases.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List recorded evaluation runs without running a new evaluation.",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Show questions and documents that fail most often in history.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        help="Compare two run indexes or run IDs.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of history items to show for --show-failures.",
    )
    parser.add_argument(
        "--history-path",
        default=None,
        help="Override the evaluation JSONL history path.",
    )
    args = parser.parse_args(argv)

    if args.list_runs or args.show_failures or args.compare:
        runs = load_evaluation_history(args.history_path)
        if args.list_runs:
            for index, run in enumerate(runs):
                metrics = run.get("metrics", {})
                print(
                    f"{index}: {run.get('run_id', '?')} {run.get('timestamp', '?')} "
                    f"{run.get('type', 'unknown')} "
                    f"{metrics.get('passed', 0)}/{metrics.get('total', 0)} passed "
                    f"Recall={metrics.get('recall_percentage', metrics.get('term_completeness_percentage'))}%"
                )
        if args.show_failures:
            for failure in summarize_failures(runs, limit=max(args.top, 1)):
                print(
                    f"{failure['count']}x | {failure['source'] or 'unsupported'} | "
                    f"{failure['question']}"
                )
        if args.compare:
            _, left = _select_history_run(runs, args.compare[0])
            _, right = _select_history_run(runs, args.compare[1])
            print(f"Comparing {left.get('run_id')} -> {right.get('run_id')}")
            for name, values in compare_evaluation_runs(left, right).items():
                print(
                    f"{name}: {values['left']} -> {values['right']} "
                    f"(delta {values['delta']})"
                )
        return runs

    from rag.retriever import retrieve

    if args.full:
        from rag.generator import MODEL, generate_answer
        from rag.prompt_builder import build_context

        report = run_answer_evaluation(
            retrieve,
            build_context,
            generate_answer,
            limit=args.limit,
            model=MODEL,
        )
        record = record_evaluation_run(report, run_type="full", history_path=args.history_path)
        print(f"Full answer checks passed: {report['passed']}/{report['total']}")
        print(f"Needs review: {report['needs_review']}")
        print(f"Term completeness: {report['term_completeness_percentage']}%")
        print(f"Citation validity: {report['citation_validity_percentage']}%")
        print(f"List completeness: {report['list_completeness_percentage']}%")
        print(f"Unsupported claims: {report['unsupported_claims']}")
        print(f"Truncated responses: {report['truncated_responses']}")
        print(f"Cache hits: {report['cache_hits']}; API calls: {report['api_calls']}")
        print(f"Run ID: {record['run_id']}")
        return report

    report = run_evaluation(retrieve)
    record = record_evaluation_run(report, history_path=args.history_path)
    print(f"Evidence checks passed: {report['passed']}/{report['total']}")
    print(f"Needs review: {report['needs_review']}")
    print(f"Precision@{report['k']}: {report['precision_percentage']}%")
    print(f"Capped precision: {report['capped_precision_percentage']}%")
    print(f"Context precision: {report['context_precision_percentage']}%")
    print(f"Recall@{report['k']}: {report['recall_percentage']}%")
    print(f"F1@{report['k']}: {report['f1_percentage']}%")
    print(f"Quality: {report['quality_label']}")
    print(f"Run ID: {record['run_id']}")
    return report


if __name__ == "__main__":
    main()
