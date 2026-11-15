import re


def _looks_like_pdf_heading(lines: list[str], index: int) -> bool:
    if index >= len(lines):
        return False
    line = lines[index].strip()
    if not line or line.startswith(("#", "-", "*", "+")):
        return False
    if len(line) > 80 or len(line.split()) > 12:
        return False
    if re.search(r"[.!?,:;]$", line):
        return False
    next_index = index + 1
    while next_index < len(lines) and not lines[next_index].strip():
        next_index += 1
    if next_index >= len(lines):
        return False
    next_line = lines[next_index].strip()
    return len(next_line.split()) >= 5 or bool(re.search(r"[.!?]$", next_line))


def _section_title(lines: list[str], plain_pdf: bool = False) -> str:
    """Derive a short human-readable label for a section."""
    if not lines:
        return ""

    first = lines[0].strip()
    if first.startswith("#"):
        return first.lstrip("#").strip()

    if plain_pdf and (
        _looks_like_pdf_heading(lines, 0)
        or (len(first) <= 80 and not re.search(r"[.!?,:;]$", first))
    ):
        return first

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
    plain_pdf: bool = False,
) -> list[dict]:
    """Split text into topic-aware chunks with lightweight metadata.

    Sections remain intact when possible. Larger sections are split by words
    with overlap so relevant facts near a boundary remain searchable.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    normalized_text = text.strip()
    if plain_pdf:
        lines = normalized_text.splitlines()
        split_lines = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped and index > 0 and _looks_like_pdf_heading(lines, index):
                if split_lines and split_lines[-1] != "":
                    split_lines.append("")
            split_lines.append(stripped)
        normalized_text = "\n".join(split_lines)

    raw_sections = [
        section.strip()
        for section in re.split(r"\n\s*\n", normalized_text)
        if section.strip()
    ]

    sections: list[tuple[str, str]] = []
    index = 0
    while index < len(raw_sections):
        section = raw_sections[index]
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        title = _section_title(lines, plain_pdf=plain_pdf)

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
