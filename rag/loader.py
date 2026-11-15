from dataclasses import dataclass, field
from hashlib import sha256
import logging
from pathlib import Path
from typing import Iterable
import re

from pypdf import PdfReader

logger = logging.getLogger(__name__)

@dataclass
class Document:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
PDF_CLEANUP_VERSION = "pdf-cleanup-v1"


def load_documents(paths: str | Path | Iterable[str | Path]) -> list[Document]:
    """Load text and PDF files into documents with stable IDs.

    Directories are scanned recursively. Empty files/documents are skipped with
    a warning. Unsupported file types raise ``ValueError``.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        else:
            files.append(path)

    documents: list[Document] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type '{suffix or '<none>'}' for '{path}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
            )
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")

        text = _read_text(path) if suffix != ".pdf" else _read_pdf(path)
        if not text.strip():
            logger.warning("Skipping empty document: %s", path)
            continue

        source = str(path.resolve())
        document_id = sha256(source.encode("utf-8")).hexdigest()
        documents.append(
            Document(
                id=document_id,
                text=text,
                source=source,
                metadata={"source": source, "file_name": path.name},
            )
        )
    return documents
    

def _read_text(path: Path) -> str:
    data = path.read_bytes()

    # Uploaded text files commonly come from Windows editors and may be
    # UTF-8, UTF-8 with BOM, UTF-16, or Windows-1252. Try strict decoding in
    # that order so valid text is preserved without silently dropping bytes.
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252")
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    # cp1252 can decode every byte, so this is defensive for unusual Python
    # codec failures and keeps the error tied to the document path.
    raise UnicodeDecodeError(
        "text", data, 0, len(data), f"Unable to decode text file: {path}"
    )


def _read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    raw_pages = [page.extract_text() or "" for page in reader.pages]
    cleaned_pages = []
    page_lines = []
    repeated_lines = {}

    for page_number, raw_page in enumerate(raw_pages, start=1):
        non_printable = sum(
            not character.isprintable() and character not in "\r\n\t"
            for character in raw_page
        )
        non_whitespace = sum(not character.isspace() for character in raw_page)
        if len(raw_page.strip()) < 40 or (
            non_whitespace and non_printable / non_whitespace > 0.05
        ):
            logger.warning(
                "PDF extraction may be broken for %s page %s: %s characters, %.1f%% non-printable",
                path,
                page_number,
                len(raw_page.strip()),
                100 * non_printable / max(non_whitespace, 1),
            )

        lines = []
        for raw_line in raw_page.splitlines():
            line = re.sub(
                r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
                " ",
                raw_line,
            )
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                lines.append("")
                continue
            lines.append(line)
            normalized = line.casefold()
            repeated_lines[normalized] = repeated_lines.get(normalized, 0) + 1
        page_lines.append(lines)

    for lines in page_lines:
        cleaned_lines = []
        for line in lines:
            normalized = line.casefold()
            is_page_marker = bool(re.fullmatch(r"page\s+\d+", normalized))
            is_repeated_header_or_footer = (
                bool(line)
                and len(line) <= 120
                and repeated_lines.get(normalized, 0) >= 2
            )
            if is_page_marker or is_repeated_header_or_footer:
                continue
            cleaned_lines.append(line)

        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        cleaned_pages.append("\n".join(cleaned_lines))

    return "\n\n".join(page for page in cleaned_pages if page).strip()
