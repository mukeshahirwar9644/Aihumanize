import os
import uuid
import zipfile
from typing import Dict

import docx2txt
from PyPDF2 import PdfReader


def read_document(file_path: str) -> str:
    file_path_l = file_path.lower()
    try:
        if file_path_l.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            return text.strip()
        if file_path_l.endswith(".docx") or file_path_l.endswith(".doc"):
            return docx2txt.process(file_path).strip()
        return ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return f"Error reading file: {e}"


def extract_zip(zip_file, extract_to: str = "uploaded_docs") -> Dict[str, str]:
    unique_folder = os.path.join(extract_to, uuid.uuid4().hex[:8])
    os.makedirs(unique_folder, exist_ok=True)

    extracted: Dict[str, str] = {}
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for member in zip_ref.namelist():
            path = os.path.join(unique_folder, member)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if member.endswith("/"):
                continue

            with zip_ref.open(member) as src, open(path, "wb") as tgt:
                tgt.write(src.read())

            base = os.path.basename(member)
            name = os.path.splitext(base)[0].replace("_", " ").strip() or base
            extracted[name] = path

    return extracted


