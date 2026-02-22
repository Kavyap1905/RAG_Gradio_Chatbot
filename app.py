import gradio as gr
import pandas as pd
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

df_global = None

load_dotenv()

# ---------- GLOBAL STORAGE ----------
vector_db = None

# ---------- LOAD CSV ----------
def process_csv(file):
    global df_global

    df_global = pd.read_csv(file.name)

    # convert rows into text
    docs = []
    for _, row in df_global.iterrows():
        text = ", ".join([f"{col}: {row[col]}" for col in df_global.columns])
        docs.append(Document(page_content=text))

    # split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = splitter.split_documents(docs)

    # embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # vector DB
    vector_db = FAISS.from_documents(splits, embedding)

    return "✅ CSV loaded successfully!"

# ---------- CHAT FUNCTION ----------
def chat(question, history):

    global df_global

    if df_global is None:
        return "Upload CSV first."

    # Give actual table to LLM
    context = df_global.to_string(index=False)

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    prompt = f"""
You are a data analyst.

Here is the dataset:

{context}

Answer the question strictly using this data.

Question: {question}
"""

    response = llm.invoke(prompt)

    return response.content


# ---------- GRADIO UI ----------
with gr.Blocks() as demo:

    gr.Markdown("# 📊 Forecast CSV RAG Chatbot (Groq + Gradio)")

    file = gr.File(label="Upload CSV")
    upload_btn = gr.Button("Process CSV")

    chatbot = gr.ChatInterface(
        fn=chat,
        title="Ask Questions About Your Data"
    )

    upload_btn.click(process_csv, inputs=file, outputs=None)

demo.launch()