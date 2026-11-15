import json
import mimetypes
import subprocess
import sys
from email import policy
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from prompt_builder import build_context
from retriever import refresh_collection, retrieve
from generator import generate_answer
from evaluation import run_answer_evaluation, run_evaluation

ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"
RAW_DIR = ROOT / "data" / "raw"
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
INGEST_LOCK = Lock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/upload":
            self.handle_upload()
            return
        if path == "/api/evaluate/full":
            try:
                report = run_answer_evaluation(
                    retrieve, build_context, generate_answer
                )
                self._json(report)
            except Exception as error:
                self._json({"error": str(error)}, status=500)
            return
        if path != "/api/ask":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("Question cannot be empty")
            # The application chooses the best retrieval strategy internally.
            # Hybrid search combines semantic similarity and BM25 without
            # asking users to understand or select a retrieval mode.
            # Use a wider evidence window for answer generation so list and
            # multi-part questions do not lose relevant items.
            chunks = retrieve(question, k=8, method="hybrid")
            if not chunks:
                answer = "I don't have enough information to answer this question."
                sources = []
            else:
                context, sources = build_context(chunks, question=question)
                answer = generate_answer(question, context)
            self._json({
                "answer": answer,
                "sources": sources,
                "evidence": chunks,
                "search_mode": "hybrid",
            })
        except Exception as error:
            self._json({"error": str(error)}, status=500)

    def handle_upload(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                raise ValueError("No files were uploaded")
            if content_length > 25 * 1024 * 1024:
                raise ValueError("Upload is limited to 25 MB")

            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("Upload must use multipart/form-data")

            body = self.rfile.read(content_length)
            message = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
                + body
            )

            RAW_DIR.mkdir(parents=True, exist_ok=True)
            uploaded = []
            # Multipart form fields use Content-Disposition: form-data, which
            # is not always returned by email.message.iter_attachments().
            # Walk leaf parts instead so browser-uploaded files are detected
            # consistently across browsers and clients.
            for part in message.walk():
                if part.is_multipart():
                    continue
                filename = part.get_filename()
                if not filename:
                    continue
                safe_name = Path(filename).name
                extension = Path(safe_name).suffix.lower()
                if extension not in ALLOWED_EXTENSIONS:
                    raise ValueError(
                        f"Unsupported file type: {extension or '<none>'}. "
                        "Use TXT, MD, or PDF."
                    )
                destination = RAW_DIR / safe_name
                destination.write_bytes(part.get_payload(decode=True) or b"")
                uploaded.append(safe_name)

            if not uploaded:
                raise ValueError("No supported documents were uploaded")

            with INGEST_LOCK:
                process = subprocess.run(
                    [sys.executable, "ingest.py"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            output = (process.stdout + process.stderr)[-12000:]
            if process.returncode != 0:
                raise RuntimeError(output or "Document ingestion failed")

            # Ingestion runs in a child process. Reload the collection so the
            # long-running web server sees newly added or replaced documents.
            refresh_collection()

            self._json({"files": uploaded, "message": "Documents indexed successfully", "log": output})
        except Exception as error:
            self._json({"error": str(error)}, status=400)

    def do_GET(self):
        if urlparse(self.path).path == "/api/evaluate":
            try:
                self._json(run_evaluation(retrieve))
            except Exception as error:
                self._json({"error": str(error)}, status=500)
            return
        super().do_GET()

    def _json(self, body, status=200):
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    print("RAG web tester: http://localhost:8000")
    ThreadingHTTPServer(("localhost", 8000), Handler).serve_forever()
