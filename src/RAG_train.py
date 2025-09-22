"""
This script demonstrates how to create vector embeddings
on custom PDF data and build a Retrieval-based chatbot.

Process: Load -> Split -> Store -> Retrieve -> Generate
"""

from pathlib import Path
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# Define base paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # points to Book_Chatbot/
PDF_DIR = BASE_DIR / "data" / "pdfs"
BOOK_CACHE_DIR = BASE_DIR / "Book" / "sentence_transformers"
MODELS_CACHE_DIR = BASE_DIR / "models"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
OFFLOAD_DIR = BASE_DIR / "offload"

print(f"PDF Directory Path: {PDF_DIR}")

# -----------------------------
# Split documents into smaller chunks (for embeddings)
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# -----------------------------
# Load PDFs and split into pages/chunks
# -----------------------------
loaders = [PyPDFLoader(str(pdf)) for pdf in PDF_DIR.glob("*.pdf")]
pages = []
for loader in loaders:
    pages.extend(loader.load_and_split(text_splitter=text_splitter))

print(f"Total chunks created from PDFs: {len(pages)}")

# -----------------------------
# Generate embeddings using HuggingFace
# -----------------------------
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=str(BOOK_CACHE_DIR),
    model_kwargs={"device": "cpu"},  # change to "cuda" if GPU is available
    encode_kwargs={"normalize_embeddings": False},
    multi_process=False,
)

# -----------------------------
# Store embeddings in Chroma vector DB
# -----------------------------
chroma_db = Chroma.from_documents(
    documents=pages,
    embedding=embed_model,
    persist_directory=str(CHROMA_DB_DIR)
)

# -----------------------------
# Setup retriever
# -----------------------------
retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5},
)

# -----------------------------
# Load TinyLlama model for Q&A
# -----------------------------
llm_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    trust_remote_code=True,
    device_map="auto",    # CPU = -1, GPU = 0
    return_full_text=False,
    temperature=0.1,
    max_new_tokens=300,
    model_kwargs={
        "cache_dir": str(MODELS_CACHE_DIR),
        "offload_folder": str(OFFLOAD_DIR),
    },
)

hf_llm = HuggingFacePipeline(pipeline=llm_pipeline)

# -----------------------------
# Strict PDF-grounded prompt
# -----------------------------
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

# -----------------------------
# Create RetrievalQA for custom PDF data
# -----------------------------
retrievalQA = RetrievalQA.from_chain_type(
    llm=hf_llm,
    retriever=retriever,
    chain_type="stuff",   # Stuff retrieved docs into the prompt
    chain_type_kwargs={"prompt": qa_prompt}
)

print("RetrievalQA ready ✅")

# -----------------------------
# Ask question using PDF data
# -----------------------------
query = "Explain the key concepts of Large Language Models from the LLM PDF"

# Uncomment for debugging (to see retrieved docs)
# docs = retriever.invoke(query)
# for i, d in enumerate(docs, 1):
#     print(f"\n--- Retrieved Doc {i} ---\n{d.page_content[:500]}...\n")

ans = retrievalQA.invoke(query)

# RetrievalQA returns a dict with "result" key
if isinstance(ans, dict) and "result" in ans:
    print("\n📌 RetrievalQA Answer:\n")
    print(ans["result"].strip())
else:
    print("\n📌 RetrievalQA Answer:\n")
    print(ans)
