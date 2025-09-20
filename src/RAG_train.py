import gradio as gr
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma  # updated import to avoid LangChain deprecation
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# -----------------------------
# Define project directories
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # points to Book_Chatbot/
PDF_DIR = BASE_DIR / "data" / "pdfs"
BOOK_CACHE_DIR = BASE_DIR / "Book" / "sentence_transformers"
MODELS_CACHE_DIR = BASE_DIR / "models"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
OFFLOAD_DIR = BASE_DIR / "offload"

# -----------------------------
# Load embeddings
# -----------------------------
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # use "cuda" if GPU is available
    encode_kwargs={"normalize_embeddings": False},
    cache_folder=str(BOOK_CACHE_DIR),
    multi_process=False,
)

# -----------------------------
# Load Chroma vector database
# -----------------------------
chroma_db = Chroma(
    persist_directory=str(CHROMA_DB_DIR),
    embedding_function=embed_model
)

# -----------------------------
# Setup retriever
# -----------------------------
retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2},  # reduce for faster response
)

# -----------------------------
# Load TinyLlama model
# -----------------------------
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

tinyllama_pipeline = pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",   # CPU = -1, GPU = 0
        dtype="auto",        # replaced deprecated torch_dtype
        cache_dir=str(MODELS_CACHE_DIR),
    ),
    tokenizer=tokenizer,
    max_new_tokens=256,     # smaller for faster inference
    temperature=0.7,
    do_sample=True,
)

tinyllama_pipeline_hf = HuggingFacePipeline(pipeline=tinyllama_pipeline)

# -----------------------------
# Setup RetrievalQA
# -----------------------------
retrievalQA = RetrievalQA.from_llm(
    llm=tinyllama_pipeline_hf,
    retriever=retriever
)

# -----------------------------
# Clean repeated lines in output
# -----------------------------
def clean_answer(ans: str) -> str:
    lines = ans.split("\n")
    seen = set()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            cleaned.append(line)
            seen.add(line)
    return "\n".join(cleaned)

# -----------------------------
# Chatbot function
# -----------------------------
def chatbot(input_text: str) -> str:
    ans = retrievalQA.invoke(input=input_text)
    return clean_answer(ans["result"])

# -----------------------------
# Gradio interface
# -----------------------------
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=7, label="Enter your text"),
    outputs="text",
    title="Information Retrieval Bot (TinyLlama)",
)

if __name__ == "__main__":
    iface.launch(share=True)