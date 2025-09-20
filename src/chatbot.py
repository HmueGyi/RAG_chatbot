"""
This script demonstrates how to create vector embeddings
on custom PDF data and build a Retrieval-based chatbot.

Process: Load -> Split -> Store -> Retrieve -> Generate
References:
- https://python.langchain.com/docs/use_cases/question_answering/
- https://python.langchain.com/docs/modules/chains/#legacy-chains
"""

from pathlib import Path
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, pipeline
from langchain.prompts import ChatPromptTemplate
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
    chunk_size=800,      # smaller chunk to avoid exceeding token limits
    chunk_overlap=200,   # overlap to maintain context
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
    search_type="mmr",         # Maximum Marginal Relevance
    search_kwargs={"k": 5},    # fewer docs to reduce token overflow
)

# -----------------------------
# Load Dolly (TinyLlama) model for Q&A
# -----------------------------
dolly_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    trust_remote_code=True,
    device_map="auto",          # CPU = -1, GPU = 0
    return_full_text=True,
    temperature=0.1,
    max_new_tokens=1000,
    model_kwargs={
        "cache_dir": str(MODELS_CACHE_DIR),
        "offload_folder": str(OFFLOAD_DIR),  # for large models
    },
)

dolly_pipeline_hf = HuggingFacePipeline(pipeline=dolly_pipeline)

# -----------------------------
# Setup prompt and output parser
# -----------------------------
question_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Question: {question}
"""
prompt_template = ChatPromptTemplate.from_template(question_template)
output_parser = StrOutputParser()
chain_1 = prompt_template | dolly_pipeline_hf | output_parser

# -----------------------------
# Test direct question (without retrieval)
# -----------------------------
chain_1_ans = chain_1.invoke(
    input={"question": "Explain the key concepts of Large Language Models from the LLM PDF"}
)
print("Chain 1 Answer:\n", chain_1_ans)

# -----------------------------
# Create RetrievalQA for custom PDF data
# -----------------------------
retrievalQA = RetrievalQA.from_llm(
    llm=dolly_pipeline_hf,
    retriever=retriever
)
print("RetrievalQA Setup:\n", retrievalQA)

# -----------------------------
# Ask question using PDF data
# -----------------------------
ans = retrievalQA.invoke(
    "Explain the key concepts of Large Language Models from the LLM PDF"
)
print("RetrievalQA Answer:\n", ans)
