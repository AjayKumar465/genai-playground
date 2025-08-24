from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
import gradio as gr
from langchain_community.llms import HuggingFacePipeline

# 1. Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# 2. Split text into chunks
def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# 3. Create vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# 4. Load a local text generation model
def load_local_llm():
    summarizer = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",   # Small & fast; you can try flan-t5-large if you want better quality
        tokenizer="google/flan-t5-base"
    )
    return summarizer

# 5. Ask questions about the PDF
def pdf_qa(file, question):
    documents = load_pdf(file.name)
    chunks = split_docs(documents)
    vectorstore = create_vector_store(chunks)

    retriever = vectorstore.as_retriever()

    llm_pipeline =  HuggingFacePipeline(pipeline=load_local_llm())

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_pipeline,
        retriever=retriever,
        chain_type="stuff"
    )

    answer = qa_chain.run(question)
    return answer

# 6. Gradio UI
iface = gr.Interface(
    fn=pdf_qa,
    inputs=[gr.File(type="file"), gr.Textbox(label="Ask a Question")],
    outputs="text",
    title="Local PDF Q&A (No API Token Required)"
)

iface.launch()
