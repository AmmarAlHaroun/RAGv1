from dataclasses import dataclass, field
from hashlib import sha256
import logging
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

logger = logging.getLogger(__name__)

@dataclass
class Document:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


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
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
