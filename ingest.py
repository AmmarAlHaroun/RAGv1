
from pathlib import Path
from hashlib import sha256
import json

import chromadb
from sentence_transformers import SentenceTransformer
from rag.chunker import chunk_text_with_metadata
from dotenv import load_dotenv
from rag.loader import PDF_CLEANUP_VERSION, load_documents
import os

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# 1. Configuration
def project_path(env_name, default):
    path = Path(os.getenv(env_name) or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


RAW_DOCS = project_path("RAG_DOCS_DIR", "data/raw")
PROCESSED_DOCS = project_path("RAG_PROCESSED_DIR", "data/processed")
STORAGE = project_path("RAG_STORAGE_DIR", "storage")
STORAGE.mkdir(exist_ok=True)
PROCESSED_DOCS.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = STORAGE / "manifest.json"

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "180"))
OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "30"))

CONFIG = {
    "model": MODEL_NAME,
    "chunk_size": CHUNK_SIZE,
    "overlap": OVERLAP,
    "collection": os.getenv("RAG_COLLECTION", "rag_documents_v1"),
    "processed_dir": str(PROCESSED_DOCS),
    "text_cleanup": PDF_CLEANUP_VERSION,
}

# 2. Open the persistent database
client = chromadb.PersistentClient(
    path=str(STORAGE / "chroma")
)

collection = client.get_or_create_collection(
    name=os.getenv("RAG_COLLECTION", "rag_documents_v1")
)
if not RAW_DOCS.is_dir():
    raise FileNotFoundError(f"Missing documents directory: {RAW_DOCS}")

# 3. Read the processing history
if MANIFEST_PATH.exists():
    manifest = json.loads(MANIFEST_PATH.read_text())
else:
    manifest = {"config": CONFIG, "files": {}}

if manifest["config"] != CONFIG:
    print("Configuration changed; rebuilding the generic document index.")
    existing_ids = collection.get().get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)
    manifest = {"config": CONFIG, "files": {}}

model = None
current_files = set()


def document_files():
    """Yield raw files first, then already processed files.

    The logical source stays relative to either folder, so moving a file
    from raw to processed does not create a duplicate index entry.
    """
    seen = set()
    for folder, is_new in ((RAW_DOCS, True), (PROCESSED_DOCS, False)):
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
                continue
            source = path.relative_to(folder).as_posix()
            if source in seen:
                continue
            seen.add(source)
            yield path, source, is_new


def move_to_processed(path, source):
    destination = PROCESSED_DOCS / source
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Replace an older processed copy when a raw document with the same
    # name is re-submitted.
    path.replace(destination)
    print("MOVED:", source, "->", destination.as_posix())

# 4. Process documents
for path, source, is_new in document_files():
    current_files.add(source)

    file_hash = sha256(path.read_bytes()).hexdigest()
    old = manifest["files"].get(source)

    # Key optimization: skip unchanged documents!
    if old and old["hash"] == file_hash:
        print("SKIPPED:", source)
        if is_new:
            move_to_processed(path, source)
        continue

    loaded = load_documents(path)
    if not loaded:
        print("SKIPPED EMPTY:", source)
        continue
    text = loaded[0].text

    chunk_records = chunk_text_with_metadata(
        text,
        CHUNK_SIZE,
        OVERLAP,
        plain_pdf=path.suffix.lower() == ".pdf",
    )
    chunks = [record["text"] for record in chunk_records]

    ids = [
        sha256(f"{source}:{i}".encode()).hexdigest()
        for i in range(len(chunks))
    ]

    if chunks:
        # Load the model only when embedding is needed
        if model is None:
            model = SentenceTransformer(MODEL_NAME)

        vectors = model.encode(
            chunks,
            normalize_embeddings=True
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=vectors,
            metadatas=[
                {
                    "source": source,
                    "file_name": path.name,
                    "file_type": path.suffix.lower().lstrip("."),
                    "chunk_index": i,
                    "section_title": record["section_title"],
                    "word_start": record["word_start"],
                    "word_end": record["word_end"],
                }
                for i, record in enumerate(chunk_records)
            ]
        )

    # Delete obsolete chunks from the old version
    old_ids = set(old["ids"]) if old else set()
    stale_ids = list(old_ids - set(ids))

    if stale_ids:
        collection.delete(ids=stale_ids)

    manifest["files"][source] = {
        "hash": file_hash,
        "ids": ids
    }

    print("INDEXED:", source, len(chunks), "chunks")
    if is_new:
        move_to_processed(path, source)

# 5. Remove records for deleted documents
for source in list(manifest["files"]):
    if source not in current_files:
        old_ids = manifest["files"][source]["ids"]
        if old_ids:
            collection.delete(ids=old_ids)

        del manifest["files"][source]
        print("REMOVED:", source)

# 6. Save processing history
temp_path = MANIFEST_PATH.with_suffix(".tmp")
temp_path.write_text(json.dumps(manifest, indent=2))
temp_path.replace(MANIFEST_PATH)

print("Total stored chunks:", collection.count())
