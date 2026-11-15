import re


def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []

    # Keep blank-line-separated sections together so one search result
    # represents one topic instead of a mixture of unrelated topics.
    raw_sections = [
        section.strip()
        for section in re.split(r"\n\s*\n", text.strip())
        if section.strip()
    ]

    sections = []
    index = 0
    while index < len(raw_sections):
        section = raw_sections[index]

        # Keep a document title with the first content section.
        if section.startswith("#") and index + 1 < len(raw_sections):
            section = f"{section}\n{raw_sections[index + 1]}"
            index += 1

        sections.append(section)
        index += 1

    for section in sections:
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        normalized_section = "\n".join(lines)
        words = normalized_section.split()

        if len(words) <= chunk_size:
            chunks.append(normalized_section)
            continue

        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))

            if end == len(words):
                break

            start = max(0, end - overlap)

    return chunks
