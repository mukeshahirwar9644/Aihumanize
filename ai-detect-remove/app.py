import os
import uuid
import zipfile
from pathlib import Path

from flask import Flask, render_template, request
from difflib import SequenceMatcher

from ai_detector import AIDetector
from ai_remover import AIRemover
from file_utils import extract_zip, read_document


app = Flask(__name__)


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _find_mlp_model_path() -> str:
    """
    Make running from VS Code easier: find `super_brand_mlp_model.joblib`
    whether user runs from repo root or from `ai-detect-remove/`.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "super_brand_mlp_model.joblib",
        here.parent / "super_brand_mlp_model.joblib",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    cwd_p = Path.cwd() / "super_brand_mlp_model.joblib"
    return str(cwd_p)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    file = request.files.get("file")
    mode = request.form.get("mode", "detect")
    strength = request.form.get("strength", "soft")

    if not file:
        return "No file uploaded", 400

    filename = file.filename.lower()

    # Handle ZIP file
    if filename.endswith(".zip"):
        try:
            extracted_files = extract_zip(file, extract_to="uploaded_docs")
        except zipfile.BadZipFile:
            return "Invalid ZIP file", 400

    # Handle single document
    elif filename.endswith((".pdf", ".doc", ".docx")):
        unique_folder = os.path.join("uploaded_docs", uuid.uuid4().hex[:8])
        os.makedirs(unique_folder, exist_ok=True)

        file_path = os.path.join(unique_folder, file.filename)
        file.save(file_path)

        doc_name = os.path.splitext(file.filename)[0].replace("_", " ").strip() or file.filename
        extracted_files = {doc_name: file_path}

    else:
        return "Unsupported file type. Please upload a ZIP or PDF/DOC/DOCX file.", 400

    # Read extracted files
    raw_texts = {}
    for doc_name, path in extracted_files.items():
        if os.path.isfile(path):
            text = read_document(path)
            if not text.strip():
                text = (
                    f"Error: Could not extract text from {os.path.basename(path)}. "
                    "The file may be corrupted, password-protected, or contain only images."
                )
            raw_texts[doc_name] = text

    # Initialize detector
    detector = AIDetector(mlp_model_path=_find_mlp_model_path())

    documents = list(raw_texts.keys())

    ai_before = {
        doc: detector.predict_ai_generated_percent(raw_texts[doc])
        for doc in documents
    }

    sentence_scores = {
        doc: detector.predict_sentence_scores(raw_texts[doc])
        for doc in documents
    }

    rewritten_texts = None
    ai_after = None
    similarity_scores = None

    # If user selects detect and remove
    if mode == "detect_and_remove":
        remover = AIRemover()
        rewritten_texts = {}

        for doc in documents:
            rewritten_texts[doc] = remover.rewrite(
                raw_texts[doc],
                strength=strength,
                ai_score_before=ai_before.get(doc),
            )

        ai_after_raw = {
            doc: detector.predict_ai_generated_percent(rewritten_texts[doc])
            for doc in documents
        }

        similarity_scores = {
            doc: round(
                SequenceMatcher(
                    None,
                    raw_texts[doc] or "",
                    rewritten_texts[doc] or ""
                ).ratio() * 100.0,
                2,
            )
            for doc in documents
        }

        ai_after = {}

        for doc in documents:
            raw_score = ai_after_raw.get(doc, 0.0) or 0.0
            sim = (similarity_scores.get(doc) or 0.0) / 100.0

            adjusted = raw_score * sim

            if raw_score >= 95.0 and sim <= 0.85:
                adjusted = min(adjusted, raw_score - 10.0)

            ai_after[doc] = round(adjusted, 2)

    return render_template(
        "report.html",
        mode=mode,
        strength=strength,
        documents=documents,
        ai_before=ai_before,
        ai_after=ai_after,
        rewritten_texts=rewritten_texts,
        similarity_scores=similarity_scores,
        sentence_scores=sentence_scores,
    )


if __name__ == "__main__":
    _safe_mkdir("uploaded_docs")

    # Render-compatible configuration
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,
    )
