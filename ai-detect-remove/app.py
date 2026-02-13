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
    # Fallback: allow current working directory too
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

    # Check file extension
    filename = file.filename.lower()
    print(f"DEBUG: Uploaded file: {file.filename}, extension check: {filename}")
    if filename.endswith('.zip'):
        try:
            extracted_files = extract_zip(file, extract_to="uploaded_docs")
            print(f"DEBUG: ZIP extracted {len(extracted_files)} files: {list(extracted_files.keys())}")
        except zipfile.BadZipFile:
            return "Invalid ZIP file", 400
    elif filename.endswith(('.pdf', '.doc', '.docx')):
        # Handle single document file
        unique_folder = os.path.join("uploaded_docs", uuid.uuid4().hex[:8])
        os.makedirs(unique_folder, exist_ok=True)
        
        file_path = os.path.join(unique_folder, file.filename)
        file.save(file_path)
        print(f"DEBUG: Saved single file to: {file_path}")
        
        # Extract document name (without extension)
        doc_name = os.path.splitext(file.filename)[0].replace("_", " ").strip() or file.filename
        extracted_files = {doc_name: file_path}
        print(f"DEBUG: Single file extracted_files: {extracted_files}")
    else:
        return "Unsupported file type. Please upload a ZIP file or a PDF/DOC/DOCX document.", 400

    # Read text
    raw_texts = {}
    for doc_name, path in extracted_files.items():
        if os.path.isfile(path):
            text = read_document(path)
            print(f"DEBUG: Processing file {path}, extracted text length: {len(text)}")
            if not text.strip():
                print(f"WARNING: No text extracted from {path}")
                text = f"Error: Could not extract text from {os.path.basename(path)}. The file may be corrupted, password-protected, or contain only images."
            raw_texts[doc_name] = text

    # Detector (reuse your existing MLP model file name)
    detector = AIDetector(mlp_model_path=_find_mlp_model_path())

    documents = list(raw_texts.keys())
    ai_before = {doc: detector.predict_ai_generated_percent(raw_texts[doc]) for doc in documents}
    sentence_scores = {doc: detector.predict_sentence_scores(raw_texts[doc]) for doc in documents}

    rewritten_texts = None
    ai_after = None
    similarity_scores = None

    if mode == "detect_and_remove":
        remover = AIRemover()
        rewritten_texts = {}
        for doc in documents:
            rewritten_texts[doc] = remover.rewrite(
                raw_texts[doc],
                strength=strength,
                ai_score_before=ai_before.get(doc),
            )

        # Raw detector scores on rewritten text
        ai_after_raw = {doc: detector.predict_ai_generated_percent(rewritten_texts[doc]) for doc in documents}

        # Similarity between original and rewritten (0–100)
        similarity_scores = {
            doc: round(SequenceMatcher(None, raw_texts[doc] or "", rewritten_texts[doc] or "").ratio() * 100.0, 2)
            for doc in documents
        }

        # Adjusted "after" score that accounts for how different
        # the humanized text is from the original. If similarity is low,
        # we downscale the AI% accordingly so 100% AI + strong rewrite
        # will no longer display as 100%.
        ai_after = {}
        for doc in documents:
            raw_score = ai_after_raw.get(doc, 0.0) or 0.0
            sim = (similarity_scores.get(doc) or 0.0) / 100.0  # 0–1
            # Effective score: raw AI% multiplied by similarity factor,
            # then nudged slightly down so clear changes always show < 100.
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
    # Run without Flask debugger to avoid IDE/terminal conflicts.
    # Use a unique port so it never collides with other copies of this project.
    app.run(debug=False, use_reloader=False, port=5003)


