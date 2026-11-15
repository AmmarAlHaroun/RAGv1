
def build_context(chunks, question=None):
    if question:
        comprehensive_terms = (
            "list", "all", "each", "every", "divided", "breakdown",
            "requirements", "modules", "percentages", "percentage",
        )
        if any(term in question.lower() for term in comprehensive_terms):
            chunks = sorted(
                chunks,
                key=lambda chunk: (
                    chunk.get("source", ""),
                    chunk.get("chunk_index", 0),
                ),
            )

    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks, start=1):
        source_id = f"S{i}"

        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        section_title = chunk.get("section_title", "")

        context_parts.append(
            f"""
                [{source_id}]
                Source: {source}
                Chunk: {chunk.get('chunk_index', 'unknown')}
                Section: {section_title or 'General'}
                Content:
                {text}
"""
        )

        sources.append({
            "id": source_id,
            "source": source,
            "chunk_index": chunk.get("chunk_index", "unknown"),
            "section_title": section_title,
        })

    context = "\n".join(context_parts)

    return context, sources
