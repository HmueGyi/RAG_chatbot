"""
Gradio playground to test a Retrieval-based chatbot
without training or embedding PDFs again.
"""

import gradio as gr
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

# ==========================================================
# Directory Settings
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
BOOK_CACHE_DIR = BASE_DIR / "Book" / "sentence_transformers"
MODELS_CACHE_DIR = BASE_DIR / "models"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
OFFLOAD_DIR = BASE_DIR / "offload"

# ==========================================================
# Load embedding model (for retrieval, not training)
# ==========================================================
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # change to "cuda" if GPU available
    encode_kwargs={"normalize_embeddings": False},
    cache_folder=str(BOOK_CACHE_DIR),
)

# ==========================================================
# Load Chroma DB (already contains embeddings)
# ==========================================================
chroma_db = Chroma(
    persist_directory=str(CHROMA_DB_DIR),
    embedding_function=embed_model
)

retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5},
)

# ==========================================================
# Load TinyLlama LLM
# ==========================================================
llm_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    trust_remote_code=True,
    device_map="auto",       # CPU=-1, GPU=0
    return_full_text=False,
    temperature=0.1,
    max_new_tokens=400,
    model_kwargs={
        "cache_dir": str(MODELS_CACHE_DIR),
        "offload_folder": str(OFFLOAD_DIR),
    },
)

hf_llm = HuggingFacePipeline(pipeline=llm_pipeline)

# ==========================================================
# Prompt Template (force concise 5-line answers)
# ==========================================================
qa_prompt = PromptTemplate(
    template="""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, reply with "I don’t know".
Keep your answer concise and limited to 5 lines.

Context:
{context}

Question: {question}
Answer (in 5 lines):
""",
    input_variables=["context", "question"],
)

retrievalQA = RetrievalQA.from_chain_type(
    llm=hf_llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt}
)

# ==========================================================
# Chatbot Function
# ==========================================================
def chatbot(input_text: str) -> str:
    ans = retrievalQA.invoke(input_text)
    return ans["result"].strip()

# ==========================================================
# Gradio Playground
# ==========================================================
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=7, label="Enter your question"),
    outputs="text",
    title="Information Retrieval Bot",
    description="Ask questions about your PDFs. The bot will answer in 5 lines using RAG."
)

if __name__ == "__main__":
    iface.launch(share=True)
