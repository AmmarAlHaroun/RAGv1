
def build_context(chunks):
    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks, start=1):
        source_id = f"S{i}"

        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")

        context_parts.append(
            f"""
                [{source_id}]
                Source: {source}
                Chunk: {chunk.get('chunk_index', 'unknown')}
                Content:
                {text}
"""
        )

        sources.append({
            "id": source_id,
            "source": source,
            "chunk_index": chunk.get("chunk_index", "unknown")
        })

    context = "\n".join(context_parts)

    return context, sources
