import gradio as gr
import pandas as pd
import os

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_groq import ChatGroq

# ---------- LOAD ENV ----------
load_dotenv()

# ---------- GLOBAL STORAGE ----------
vector_db = None


# ---------- PROCESS CSV ----------
def process_csv(file):
    global vector_db

    df = pd.read_csv(file.name)

    docs = []

    for _, row in df.iterrows():
        text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(page_content=text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(splits, embeddings)

    return "✅ CSV processed and indexed!"


# ---------- CHAT FUNCTION ----------
def chat(message, history):

    global vector_db

    if vector_db is None:
        return "⚠️ Please upload CSV first."

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a data analyst.

Use ONLY the provided context to answer.

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(
        vector_db.as_retriever(),
        document_chain
    )

    response = retrieval_chain.invoke({"input": message})

    return response["answer"]


# ---------- GRADIO UI ----------
with gr.Blocks() as demo:

    gr.Markdown("# 📊 Forecast CSV RAG Chatbot (Groq + Gradio)")

    file = gr.File(label="Upload CSV")
    upload_btn = gr.Button("Process CSV")

    status = gr.Textbox(label="Status")

    upload_btn.click(
        process_csv,
        inputs=file,
        outputs=status
    )

    gr.ChatInterface(
        fn=chat,
        title="Ask Questions About Your Data"
    )

demo.launch()