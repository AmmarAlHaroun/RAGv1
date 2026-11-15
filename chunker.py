import re


def _section_title(lines: list[str]) -> str:
    """Derive a short human-readable label for a section."""
    if not lines:
        return ""

    first = lines[0].strip()
    if first.startswith("#"):
        return first.lstrip("#").strip()

    first = re.sub(r"^[-*+]\s*", "", first)
    if ":" in first:
        label = first.split(":", 1)[0].strip()
        if 2 <= len(label) <= 80:
            return label
    return ""


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[dict]:
    """Split text into topic-aware chunks with lightweight metadata.

    Sections remain intact when possible. Larger sections are split by words
    with overlap so relevant facts near a boundary remain searchable.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    raw_sections = [
        section.strip()
        for section in re.split(r"\n\s*\n", text.strip())
        if section.strip()
    ]

    sections: list[tuple[str, str]] = []
    index = 0
    while index < len(raw_sections):
        section = raw_sections[index]
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        title = _section_title(lines)

        # Keep a document title with its first content section.
        if lines and lines[0].startswith("#") and index + 1 < len(raw_sections):
            next_lines = [
                line.strip()
                for line in raw_sections[index + 1].splitlines()
                if line.strip()
            ]
            lines.extend(next_lines)
            index += 1

        normalized_section = "\n".join(lines)
        sections.append((normalized_section, title))
        index += 1

    chunks: list[dict] = []
    for section, title in sections:
        words = section.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append({
                "text": " ".join(words[start:end]),
                "section_title": title,
                "word_start": start,
                "word_end": end,
            })

            if end == len(words):
                break
            start = end - overlap

    return chunks


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> list[str]:
    """Backward-compatible text-only chunking helper."""
    return [
        chunk["text"]
        for chunk in chunk_text_with_metadata(text, chunk_size, overlap)
    ]
