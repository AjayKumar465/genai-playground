# pdf_chatbot_v3.py
# Runs fully local, avoids token overflow, and returns detailed answers.

import os
import gradio as gr
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# 0) Model + tokenizer (load once)
# -----------------------------
MODEL_NAME = "google/flan-t5-large"   # if too slow on your Mac, switch to "google/flan-t5-base"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

LLM_PIPELINE = pipeline(
    task="text2text-generation",
    model=MODEL_NAME,
    tokenizer=TOKENIZER,
    # generation controls
    max_new_tokens=256,   # length of the answer
    do_sample=True,
    temperature=0.7,
    num_beams=4
)

# -----------------------------
# 1) Globals for vectorstore (kept in memory after processing)
# -----------------------------
VECTORSTORE = None
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# 2) Token-aware splitter (uses the HF tokenizer)
# -----------------------------
def make_splitter():
    try:
        # splits by *token count* using this tokenizer
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            TOKENIZER,
            chunk_size=256,     # <= 256 input tokens per chunk
            chunk_overlap=32
        )
        return splitter
    except Exception:
        # Fallback (char-based). Still OK, but less precise.
        from langchain.text_splitter import CharacterTextSplitter
        return CharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# -----------------------------
# 3) Build vector index from PDF
# -----------------------------
def process_pdf(pdf_file):
    global VECTORSTORE
    if pdf_file is None:
        return "Please upload a PDF first."

    loader = PyPDFLoader(pdf_file.name)
    docs = loader.load()

    splitter = make_splitter()
    chunks = splitter.split_documents(docs)

    VECTORSTORE = FAISS.from_documents(chunks, EMBEDDINGS)
    return f"✅ Indexed {len(chunks)} chunks. You can ask questions now."

# -----------------------------
# 4) Token-safe context builder
# -----------------------------
MAX_INPUT_TOKENS = 480   # keep below 512; leave room for prompt+question
PROMPT_HEADER = (
    "You are a helpful assistant.\n"
    "Answer the question ONLY using the context below. "
    "If the answer is not present in the context, say you do not know.\n\n"
    "Context:\n"
)

def tokens_len(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def build_context_with_limit(chunks: List[str], question: str) -> str:
    """
    Packs as many chunk texts as will fit under MAX_INPUT_TOKENS budget
    (together with prompt and question). Truncates the last chunk if needed.
    """
    # rough prompt + question token cost
    prompt_q = f"{PROMPT_HEADER}{{context}}\n\nQuestion:\n{question}\n\nAnswer:"
    prompt_tokens = tokens_len(prompt_q.replace("{context}", ""))

    budget = MAX_INPUT_TOKENS - prompt_tokens
    if budget < 100:
        budget = 100  # minimum safety

    selected = []
    used = 0

    for text in chunks:
        t = tokens_len(text)
        if used + t <= budget:
            selected.append(text)
            used += t
        else:
            remaining = budget - used
            if remaining <= 0:
                break
            # truncate last chunk to fit the remaining token budget
            ids = TOKENIZER.encode(text, add_special_tokens=False)
            trimmed = TOKENIZER.decode(ids[:remaining])
            if trimmed.strip():
                selected.append(trimmed)
            break

    return "\n\n".join(selected)

# -----------------------------
# 5) Answer function (retrieval + controlled prompt)
# -----------------------------
def answer_question(question):
    if not question or question.strip() == "":
        return "Please enter a question."

    if VECTORSTORE is None:
        return "Please process a PDF first."

    # Get a few most relevant chunks (their raw text)
    retriever = VECTORSTORE.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)
    chunk_texts = [d.page_content for d in docs]

    # Build token-budgeted context
    context = build_context_with_limit(chunk_texts, question)

    # Final prompt
    prompt = (
        f"{PROMPT_HEADER}"
        f"{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer in a detailed, step-by-step manner:"
    )

    # Generate
    out = LLM_PIPELINE(prompt)
    # HF pipelines return a list of dicts with 'generated_text'
    return out[0]["generated_text"]

# -----------------------------
# 6) Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Local PDF Q&A (Token-Safe, Detailed Answers)")

    with gr.Row():
        with gr.Column():
            pdf = gr.File(label="Upload PDF", type="file")
            build_btn = gr.Button("Process PDF")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            q = gr.Textbox(label="Ask a question", lines=2)
            ask_btn = gr.Button("Answer")
            a = gr.Textbox(label="Answer (detailed)", lines=10)

    build_btn.click(process_pdf, inputs=pdf, outputs=status)
    ask_btn.click(answer_question, inputs=q, outputs=a)

if __name__ == "__main__":
    demo.launch()
