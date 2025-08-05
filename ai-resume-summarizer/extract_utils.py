import os
import docx2txt
import PyPDF2
from langdetect import detect
from deep_translator import GoogleTranslator
from collections import Counter
import re


def extract_text_from_file(file_obj):
    if file_obj.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file_obj)
        return " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    elif file_obj.name.endswith(".docx"):
        return docx2txt.process(file_obj.name)
    else:
        return ""

def translate_to_english(text):
    lang = detect(text)
    if lang != "en":
        return GoogleTranslator(source=lang, target="en").translate(text)
    return text

def extract_skills(text):
    common_skills = ["python", "java", "sql", "aws", "docker", "kubernetes", "react", "spring"]
    found = [skill for skill in common_skills if skill.lower() in text.lower()]
    return found

def extract_experience(text):
    match = re.search(r'(\d+)\+?\s+years?', text.lower())
    return match.group(1) if match else "Not mentioned"

def extract_title(text):
    for line in text.splitlines():
        if "engineer" in line.lower() or "developer" in line.lower():
            return line.strip()
    return "Not found"

def extract_qualification(text):
    for word in ["B.Tech", "M.Tech", "B.Sc", "B.E", "MCA", "MBA", "PhD"]:
        if word.lower() in text.lower():
            return word
    return "Not found"

def extract_companies(text):
    matches = re.findall(r'at\s+([A-Z][a-zA-Z0-9&\s]+)', text)
    return list(set(matches))
