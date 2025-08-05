import os
import fitz  # PyMuPDF
from docx import Document
import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from summarizer import summarize_text  # reuse your model/tokenizer setup


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def analyze_resume(file, model_choice, use_structured_prompt):
    if not file:
        return "❌ No file uploaded."

    ext = os.path.splitext(file.name)[-1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file.name)
    elif ext == ".docx":
        text = extract_text_from_docx(file.name)
    else:
        return "❌ Unsupported file format. Use PDF or DOCX."

    if use_structured_prompt:
        structured_prompt = f"""
        Analyze this resume and extract:
        - Top 5 Skills
        - Top 3 Companies Worked At
        - Total Years of Experience (if available)
        - Most Recent Job Title
        - Highest Qualification

        Resume:
        {text}
        """
        return summarize_text(structured_prompt, model_choice)
    else:
        return summarize_text(text, model_choice)
