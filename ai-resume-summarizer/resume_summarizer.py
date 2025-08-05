import os
import fitz  # PyMuPDF
from docx import Document
import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from summarizer import summarize_text  # reuse your model/tokenizer setup
from tempfile import NamedTemporaryFile

from collections import Counter
import matplotlib.pyplot as plt
import json
from tempfile import NamedTemporaryFile

from summarizer import summarize_text
from extract_utils import (
    extract_text_from_file,
    translate_to_english,
    extract_skills,
    extract_experience,
    extract_title,
    extract_qualification,
    extract_companies
)


def plot_skills(skills):
    if not skills:
        return None
    count = Counter(skills)
    plt.figure(figsize=(6, 3))
    plt.bar(count.keys(), count.values())
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_path = "skills_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

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
    

def process_resume(file, model_choice, max_len, min_len, structured_prompt):
    text = extract_text_from_file(file)
    text = translate_to_english(text)

    if structured_prompt:
        data = {
            "skills": extract_skills(text),
            "companies": extract_companies(text),
            "experience": extract_experience(text),
            "title": extract_title(text),
            "qualification": extract_qualification(text)
        }
        summary = json.dumps(data, indent=2)
        chart = plot_skills(data["skills"])
    else:
        summary = summarize_text(text, model_choice, max_len, min_len)
        chart = None

    # Save summary
    with NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as tmp:
        tmp.write(summary)
        summary_path = tmp.name

    return summary, summary_path, chart

 
