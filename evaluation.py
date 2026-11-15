import re


EVAL_CASES = [
    {
        "question": "What are the prerequisites for AI-101?",
        "source": "course.txt",
        "terms": ["prerequisites for ai-101", "basic python", "introductory statistics"],
        "answer": "Basic Python programming and introductory statistics.",
    },
    {
        "question": "What is the late submission penalty?",
        "source": "university_policy.txt",
        "terms": ["late submission rule", "10 percent", "calendar day"],
        "answer": "10 percent of the mark per calendar day, with a maximum of 3 calendar days. An approved extension removes the penalty.",
    },
    {
        "question": "How many credit hours is AI-101?",
        "source": "course.txt",
        "terms": ["credit hours", "3"],
        "answer": "3 credit hours.",
    },
    {
        "question": "What software is required?",
        "source": "course.txt",
        "terms": ["required software", "python 3.11", "chromadb", "sentence transformers"],
        "answer": "Python 3.11 or later, ChromaDB, and Sentence Transformers.",
    },
    {
        "question": "Can students use AI tools for assignments?",
        "source": "university_policy.txt",
        "terms": ["artificial intelligence rule", "ai tools", "instructor", "assignment"],
        "answer": "AI tools may be used only when the instructor or assignment allows them.",
    },
    {
        "question": "How long do students have to request a grade review?",
        "source": "university_policy.txt",
        "terms": ["grade review rule", "7 calendar days"],
        "answer": "Within 7 calendar days of receiving the grade.",
    },
    {
        "question": "What are the final project requirements?",
        "source": "course.txt",
        "terms": ["final project requirement", "two source documents", "ten questions"],
        "answer": "Use at least two source documents, searchable chunks with metadata, a vector database, at least ten test questions, and a report of results and limitations.",
    },
    {
        "question": "What background knowledge is needed before AI-101?",
        "source": "course.txt",
        "terms": ["prerequisites for ai-101", "basic python", "introductory statistics"],
        "answer": "Basic Python programming and introductory statistics.",
    },
    {
        "question": "How are late assignments handled, including extensions?",
        "source": "university_policy.txt",
        "terms": ["late submission rule", "10 percent", "approved extension"],
        "answer": "Late work loses 10 percent per calendar day; an approved extension removes the penalty.",
    },
    {
        "question": "How are the course assessment percentages divided?",
        "source": "course.txt",
        "terms": ["assessment weight", "25 percent", "20 percent", "100 percent"],
        "answer": "Programming assignments are 25%, laboratories 20%, the midterm 20%, the final project 25%, and participation 10%; the total is 100%.",
    },
    {
        "question": "What is the capital of France?",
        "source": None,
        "terms": [],
        "answer": "I don't have enough information to answer this question.",
        "should_answer": False,
    },
    {
        "question": "What is the tuition fee?",
        "source": None,
        "terms": [],
        "answer": "I don't have enough information to answer this question.",
        "should_answer": False,
    },
    {
        "question": "Explain quantum entanglement.",
        "source": None,
        "terms": [],
        "answer": "I don't have enough information to answer this question.",
        "should_answer": False,
    },
]


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return None

    retrieved = set(ranked_ids[:k])

    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)

    if k < 1:
        raise ValueError("k must be at least 1")

    retrieved = ranked_ids[:k]

    relevant_count = sum(
        chunk_id in relevant_ids
        for chunk_id in retrieved
    )

    return relevant_count / k


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant_ids = set(relevant_ids)

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0


def _source_matches(chunk, expected_source):
    return bool(
        expected_source
        and chunk.get("source", "").replace("\\", "/").endswith(expected_source)
    )


def _relevant_chunk_ids(chunks, case):
    """Build relevance labels from the expected source and answer terms."""
    expected_source = case.get("source")
    all_terms = [term.lower() for term in case.get("terms", [])]
    # Standalone numbers are too broad to label a chunk as relevant (for
    # example, many course chunks may contain the number "3").
    expected_terms = [term for term in all_terms if not term.strip().isdigit()] or all_terms
    if not expected_source or not expected_terms:
        return []

    return [
        chunk["id"]
        for chunk in chunks
        if _source_matches(chunk, expected_source)
        and any(term in chunk.get("text", "").lower() for term in expected_terms)
    ]


def _percentage(value):
    return round(value * 100, 2) if value is not None else None


def run_evaluation(retrieve_fn, k=5):
    """Check whether the top evidence chunks contain the expected answer."""
    results = []

    for case in EVAL_CASES:
        question = case["question"]
        expected_source = case["source"]
        expected_terms = case["terms"]
        should_answer = case.get("should_answer", True)
        # Retrieve the full ranked set once so recall has a corpus-wide
        # denominator while the displayed evidence still uses top-k results.
        ranked_chunks = retrieve_fn(question, k=max(k, 1000))
        chunks = ranked_chunks[:k]
        relevant_ids = _relevant_chunk_ids(ranked_chunks, case)
        ranked_ids = [chunk["id"] for chunk in ranked_chunks]
        precision = precision_at_k(ranked_ids, relevant_ids, k) if relevant_ids else None
        recall = recall_at_k(ranked_ids, relevant_ids, k) if relevant_ids else None
        top_chunk = chunks[0] if chunks else {}
        relevant_chunks = [
            chunk for chunk in chunks
            if _source_matches(chunk, expected_source)
        ]
        evidence_text = "\n".join(chunk.get("text", "") for chunk in relevant_chunks).lower()
        found_terms = [term for term in expected_terms if term.lower() in evidence_text]
        missing_terms = [term for term in expected_terms if term.lower() not in evidence_text]
        correct = (not chunks) if not should_answer else bool(relevant_chunks) and not missing_terms
        results.append({
            "question": question,
            "expected_source": expected_source,
            "expected_answer": case["answer"],
            "should_answer": should_answer,
            "expected_terms": expected_terms,
            "found_terms": found_terms,
            "missing_terms": missing_terms,
            "passed": correct,
            "top_source": top_chunk.get("source"),
            "top_chunk": top_chunk.get("chunk_index"),
            "retrieved_text": "\n".join(chunk.get("text", "") for chunk in relevant_chunks) or top_chunk.get("text", ""),
            "relevant_chunks": len(relevant_ids),
            "precision_at_k": round(precision, 4) if precision is not None else None,
            "recall_at_k": round(recall, 4) if recall is not None else None,
            "precision_percentage": _percentage(precision),
            "recall_percentage": _percentage(recall),
        })

    passed = sum(result["passed"] for result in results)
    precision_values = [
        result["precision_at_k"]
        for result in results
        if result["precision_at_k"] is not None
    ]
    recall_values = [
        result["recall_at_k"]
        for result in results
        if result["recall_at_k"] is not None
    ]
    average_precision = (
        sum(precision_values) / len(precision_values)
        if precision_values else None
    )
    average_recall = (
        sum(recall_values) / len(recall_values)
        if recall_values else None
    )

    return {
        "total": len(results),
        "passed": passed,
        "needs_review": len(results) - passed,
        "k": k,
        "precision_at_k": round(average_precision, 4) if average_precision is not None else None,
        "recall_at_k": round(average_recall, 4) if average_recall is not None else None,
        "precision_percentage": _percentage(average_precision),
        "recall_percentage": _percentage(average_recall),
        "metric_cases": len(precision_values),
        "results": results,
    }


def run_answer_evaluation(retrieve_fn, build_context_fn, generate_fn, k=5):
    """Run the slower end-to-end check, including citations in generated answers."""
    results = []
    for case in EVAL_CASES:
        question = case["question"]
        chunks = retrieve_fn(question, k=k)
        expected_terms = case["terms"]
        should_answer = case.get("should_answer", True)
        if not chunks:
            answer = "I don't have enough information to answer this question."
        else:
            context, _ = build_context_fn(chunks)
            answer = generate_fn(question, context)

        citations = re.findall(r"\[S(\d+)\]", answer)
        valid_citations = bool(citations) and all(
            1 <= int(citation) <= len(chunks) for citation in citations
        )
        answer = answer or ""
        answer_terms_found = [
            term for term in expected_terms if term.lower() in answer.lower()
        ]
        if should_answer:
            passed = bool(chunks) and len(answer_terms_found) == len(expected_terms) and valid_citations
        else:
            passed = not chunks and "don't have enough information" in answer.lower()
        results.append({
            "question": question,
            "expected_answer": case["answer"],
            "answer": answer,
            "passed": passed,
            "citations": [f"S{citation}" for citation in citations],
            "valid_citations": valid_citations,
            "answer_terms_found": answer_terms_found,
        })

    passed = sum(result["passed"] for result in results)
    return {
        "total": len(results),
        "passed": passed,
        "needs_review": len(results) - passed,
        "results": results,
    }


if __name__ == "__main__":
    from retriever import retrieve

    report = run_evaluation(retrieve)
    print(f"Evidence checks passed: {report['passed']}/{report['total']}")
    print(f"Needs review: {report['needs_review']}")
    print(f"Precision@{report['k']}: {report['precision_percentage']}%")
    print(f"Recall@{report['k']}: {report['recall_percentage']}%")
    for result in report["results"]:
        status = "CORRECT" if result["passed"] else "CHECK"
        print(f"{status}: {result['question']}")
