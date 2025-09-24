"""
Gradio RAG Chatbot:
Load existing Chroma DB (no retraining),
query with TinyLlama, and show answers with sources.
"""

import gradio as gr
from pathlib import Path
from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import torch

# -----------------------------
# Define paths (same as training script)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
BOOK_CACHE_DIR = BASE_DIR / "Book" / "sentence_transformers"
MODELS_CACHE_DIR = BASE_DIR / "models"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
OFFLOAD_DIR = BASE_DIR / "offload"

# -----------------------------
# Embeddings (must match training script)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    cache_folder=str(BOOK_CACHE_DIR),
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
    multi_process=False,
)

# -----------------------------
# Load existing Chroma DB
# -----------------------------
if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
    raise RuntimeError(f"No Chroma DB found at {CHROMA_DB_DIR}. Run the ingestion script first!")

print("Loading Chroma DB...")
chroma_db = Chroma(
    persist_directory=str(CHROMA_DB_DIR),
    collection_name="document_collection",
    embedding_function=embed_model,
)

# Use similarity search for speed
retriever = chroma_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# -----------------------------
# TinyLlama model for Q&A
# -----------------------------
print("Loading TinyLlama model...")
llm_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    trust_remote_code=True,
    device_map={"": 0} if device == "cuda" else "cpu",
    return_full_text=False,
    temperature=0.1,
    max_new_tokens=150,
    do_sample=False,
    model_kwargs={
        "cache_dir": str(MODELS_CACHE_DIR),
        "offload_folder": str(OFFLOAD_DIR),
    },
)
hf_llm = HuggingFacePipeline(pipeline=llm_pipeline)

# -----------------------------
# Prompt
# -----------------------------
qa_prompt = PromptTemplate(
    template="""
Answer the following question using ONLY the context provided.
Be concise: 1–3 sentences for simple answers, up to 8 sentences for complex answers.
Mention the source filename for each piece of information if possible.
If the answer is not in the context, reply exactly with: I don't know

Context:
{context}

Question: {question}

Answer:
""",
    input_variables=["context", "question"],
)

# -----------------------------
# RetrievalQA
# -----------------------------
retrievalQA = RetrievalQA.from_chain_type(
    llm=hf_llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
)

print("RAG Chatbot ready ✅")

# -----------------------------
# Gradio chat function
# -----------------------------
def chat(query, history):
    ans = retrievalQA.invoke(query)

    if isinstance(ans, dict) and "result" in ans:
        ans_text = ans["result"].strip()
        sources = [doc.metadata.get("source_file") for doc in ans.get("source_documents", [])]
    else:
        ans_text, sources = str(ans).strip(), []

    if not ans_text or ans_text.lower() in ["", "none", "unknown"]:
        ans_text = "I don't know"

    # Add sources
    if sources:
        ans_text += f"\n\n📂 Sources: {', '.join(set(sources))}"

    history.append((query, ans_text))
    return history, history

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📖 High-Accuracy RAG Chatbot (TinyLlama + Chroma)")
    chatbot = gr.Chatbot([], elem_id="chatbot", height=500)
    msg = gr.Textbox(placeholder="Ask a question about your documents...")
    clear = gr.Button("Clear")

    state = gr.State([])

    msg.submit(chat, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch()