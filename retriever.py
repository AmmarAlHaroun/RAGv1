import math
import os
import re
from collections import Counter

import chromadb
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore[import-not-found]

load_dotenv()

# The embedding configuration is shared with ingestion.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DB_PATH = os.path.join(os.getenv("RAG_STORAGE_DIR", "storage"), "chroma")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "rag_documents_v1")

DEFAULT_SEARCH_MODE = os.getenv("RAG_SEARCH_MODE", "hybrid").strip().lower()
HYBRID_SEMANTIC_WEIGHT = float(os.getenv("RAG_HYBRID_SEMANTIC_WEIGHT", "0.65"))
if not 0 <= HYBRID_SEMANTIC_WEIGHT <= 1:
    raise ValueError("RAG_HYBRID_SEMANTIC_WEIGHT must be between 0 and 1")
RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
RERANK_MODEL_NAME = os.getenv(
    "RAG_RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANK_CANDIDATES = max(int(os.getenv("RAG_RERANK_CANDIDATES", "10")), 1)

# Initialize the database immediately, but load the embedding model only when
# semantic or hybrid search is requested. Keyword-only queries do not need it.
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
_model = None
_reranker = None

STOP_WORDS = {
    "what", "when", "where", "which", "who", "how", "does", "do",
    "is", "are", "the", "a", "an", "of", "for", "to", "in", "on",
    "and", "or", "can", "i", "my", "with", "about", "before", "after",
    "explain", "describe", "list", "tell", "give", "show",
}

# Keep punctuation inside identifiers and technical terms, then also index
# their pieces. This lets "AI-101", "AI 101", and "python 3.11" match each
# other without throwing away the exact identifier signal.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._:/-][a-z0-9]+)*", re.IGNORECASE)
TOKEN_SEPARATOR_PATTERN = re.compile(r"[._:/-]+")


def _tokenize(value: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(value.lower()):
        tokens.append(raw_token)
        tokens.extend(
            part
            for part in TOKEN_SEPARATOR_PATTERN.split(raw_token)
            if part and part != raw_token
        )
    return tokens


def _keep_token(token: str) -> bool:
    # Keep numeric tokens such as "3" because they are often meaningful in
    # course codes, versions, percentages, and policy limits.
    return token not in STOP_WORDS and (len(token) > 1 or token.isdigit())


def _terms(value: str) -> set[str]:
    return {token for token in _tokenize(value) if _keep_token(token)}


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def _canonical_search_mode(
    method: str | None,
    mode: str | None,
    search_type: str | None,
) -> str:
    requested = method or mode or search_type or DEFAULT_SEARCH_MODE
    normalized = requested.strip().lower()
    aliases = {
        "bm25": "keyword",
        "keyword": "keyword",
        "lexical": "keyword",
        "semantic": "semantic",
        "vector": "semantic",
        "hybrid": "hybrid",
    }
    if normalized not in aliases:
        choices = "semantic, keyword (bm25), hybrid"
        raise ValueError(f"Unknown search mode '{requested}'. Choose {choices}.")
    return aliases[normalized]


def _records_from_chroma(result: dict, nested: bool = False) -> list[dict]:
    ids = result.get("ids", [])
    documents = result.get("documents", []) or []
    metadatas = result.get("metadatas", []) or []
    distances = result.get("distances", []) or []

    if nested:
        ids = ids[0] if ids else []
        documents = documents[0] if documents else []
        metadatas = metadatas[0] if metadatas else []
        distances = distances[0] if distances else []

    records = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        records.append({
            "id": chunk_id,
            "text": documents[index] if index < len(documents) else "",
            "source": (metadata or {}).get("source", "unknown"),
            "distance": distances[index] if index < len(distances) else None,
            "chunk_index": (metadata or {}).get("chunk_index", index),
        })
    return records


def _load_records(question: str, mode: str, source: str | None) -> list[dict]:
    where = {"source": source} if source is not None else None

    if mode in {"semantic", "hybrid"}:
        question_vector = _get_model().encode(
            question,
            normalize_embeddings=True,
        ).tolist()
        options = {
            "query_embeddings": [question_vector],
            "n_results": collection.count(),
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            options["where"] = where
        return _records_from_chroma(collection.query(**options), nested=True)

    options = {
        "include": ["documents", "metadatas"],
    }
    if where is not None:
        options["where"] = where
    return _records_from_chroma(collection.get(**options))


def _bm25_scores(question: str, records: list[dict]) -> list[float]:
    """Return Okapi BM25 scores for each record in the same order."""
    query_terms = _terms(question)
    if not query_terms or not records:
        return [0.0] * len(records)

    tokenized_documents = [
        [token for token in _tokenize(record["text"]) if _keep_token(token)]
        for record in records
    ]
    document_frequency = Counter(
        token
        for tokens in tokenized_documents
        for token in set(tokens)
    )
    document_count = len(tokenized_documents)
    average_length = sum(map(len, tokenized_documents)) / document_count
    if average_length == 0:
        return [0.0] * document_count

    k1 = 1.5
    b = 0.75
    scores = []
    for tokens in tokenized_documents:
        term_frequency = Counter(tokens)
        document_length = len(tokens)
        normalization = k1 * (1 - b + b * document_length / average_length)
        score = 0.0
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue
            # BM25's smoothed IDF stays positive even for common terms.
            idf = math.log(
                1 + (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            score += idf * (frequency * (k1 + 1)) / (frequency + normalization)
        scores.append(score)
    return scores


def _normalize_scores(scores: list[float]) -> list[float]:
    maximum = max(scores, default=0.0)
    if maximum <= 0:
        return [0.0] * len(scores)
    return [score / maximum for score in scores]


def _semantic_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    # The collection uses Chroma's default squared-L2 distance and embeddings
    # are normalized, so cosine similarity is 1 - (distance / 2).
    return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))


def retrieve(
    question: str,
    k: int = 3,
    source: str | None = None,
    method: str | None = None,
    mode: str | None = None,
    search_type: str | None = None,
):
    """Retrieve relevant chunks with semantic, BM25, or hybrid ranking.

    ``hybrid`` is the default and combines normalized semantic similarity with
    normalized BM25 scores. ``method``, ``mode``, and ``search_type`` are
    accepted as aliases so callers can use the name that fits their API.
    """
    if not question.strip():
        return []
    if k < 1:
        raise ValueError("k must be at least 1")
    if collection.count() == 0:
        return []

    selected_mode = _canonical_search_mode(method, mode, search_type)
    chunks = _load_records(question, selected_mode, source)
    if not chunks:
        return []

    raw_keyword_scores = _bm25_scores(question, chunks)
    keyword_scores = _normalize_scores(raw_keyword_scores)
    semantic_scores = [
        _semantic_score(chunk["distance"])
        if selected_mode in {"semantic", "hybrid"}
        else 0.0
        for chunk in chunks
    ]

    for index, chunk in enumerate(chunks):
        if selected_mode == "semantic":
            final_score = semantic_scores[index]
        elif selected_mode == "keyword":
            final_score = keyword_scores[index]
        else:
            final_score = (
                HYBRID_SEMANTIC_WEIGHT * semantic_scores[index]
                + (1 - HYBRID_SEMANTIC_WEIGHT) * keyword_scores[index]
            )
        chunk["semantic_score"] = round(semantic_scores[index], 6)
        chunk["keyword_score"] = round(keyword_scores[index], 6)
        chunk["bm25_score"] = round(raw_keyword_scores[index], 6)
        chunk["score"] = round(final_score, 6)
        chunk["search_mode"] = selected_mode

    chunks.sort(
        key=lambda chunk: (
            chunk["score"],
            chunk["keyword_score"],
            chunk["semantic_score"],
        ),
        reverse=True,
    )
    min_score = float(os.getenv("RAG_MIN_SCORE", "0.3"))
    eligible = [chunk for chunk in chunks if chunk["score"] >= min_score]

    if RERANK_ENABLED and eligible:
        candidates = eligible[:max(k, RERANK_CANDIDATES)]
        try:
            pairs = [(question, chunk["text"]) for chunk in candidates]
            rerank_scores = _get_reranker().predict(
                pairs,
                show_progress_bar=False,
            )
            for chunk, rerank_score in zip(candidates, rerank_scores):
                chunk["rerank_score"] = round(float(rerank_score), 6)
            candidates.sort(
                key=lambda chunk: (
                    chunk["rerank_score"],
                    chunk["score"],
                ),
                reverse=True,
            )
            return candidates[:k]
        except Exception as error:
            # Retrieval remains usable if the optional reranker model cannot
            # be downloaded or loaded in the current environment.
            print(f"Reranker unavailable; using hybrid ranking: {error}")

    return eligible[:k]
