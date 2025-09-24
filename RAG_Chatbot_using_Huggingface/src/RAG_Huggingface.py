"""
High-accuracy RAG Chatbot
- Loads PDFs, text, images, and Word docs
- Creates embeddings and stores them in ChromaDB
- Uses TinyLlama for retrieval-based Q&A
"""

from pathlib import Path
import re
from typing import List

from typing import List, Tuple
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, pipeline
from langchain_community.document_loaders import (
    TextLoader, UnstructuredImageLoader,
    UnstructuredWordDocumentLoader, PyPDFLoader
)

# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
DATA = {
    "pdf": BASE / "data/pdfs",
    "txt": BASE / "data/txts",
    "img": BASE / "data/images",
    "docx": BASE / "data/docs",
}
CHROMA_DB = BASE / "chroma_db"
CACHE_BOOK = BASE / "Book/sentence_transformers"
CACHE_MODELS = BASE / "models"
OFFLOAD = BASE / "offload"

# -----------------------------
# Splitter + Embeddings
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    cache_folder=str(CACHE_BOOK),
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# -----------------------------
# Load documents helper
# -----------------------------
def load_docs(path: Path, loader_cls) -> List[Document]:
    docs = []
    for file in path.glob("*.*"):
        try:
            loader = loader_cls(str(file))
            parts = loader.load_and_split(text_splitter=splitter)
            for d in parts:
                d.metadata["source_file"] = file.name
            docs.extend(parts)
        except Exception as e:
            print(f"⚠️ Failed to load {file}: {e}")
    return docs

# -----------------------------
# Chroma DB
# -----------------------------
if CHROMA_DB.exists() and any(CHROMA_DB.iterdir()):
    print("Loading existing Chroma DB...")
    chroma = Chroma(
        persist_directory=str(CHROMA_DB),
        collection_name="docs",
        embedding_function=embedder,
    )
else:
    print("Building new Chroma DB...")
    pages = []
    pages += load_docs(DATA["pdf"], PyPDFLoader)
    pages += load_docs(DATA["txt"], TextLoader)
    pages += load_docs(DATA["img"], UnstructuredImageLoader)
    pages += load_docs(DATA["docx"], UnstructuredWordDocumentLoader)

    # Deduplicate
    seen, unique = set(), []
    for p in pages:
        if p.page_content not in seen:
            unique.append(p)
            seen.add(p.page_content)

    chroma = Chroma.from_documents(
        documents=unique,
        embedding=embedder,
        persist_directory=str(CHROMA_DB),
        collection_name="docs",
    )
    print(f"✅ Chroma DB saved at {CHROMA_DB}")

# -----------------------------
# Retriever + LLM
# -----------------------------
retriever = chroma.as_retriever(search_type="mmr", search_kwargs={"k": 5})

llm_pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    device_map="auto",
    return_full_text=False,
    temperature=0.2,
    max_new_tokens=600,
    do_sample=True,
    model_kwargs={"cache_dir": str(CACHE_MODELS), "offload_folder": str(OFFLOAD)},
)
llm = HuggingFacePipeline(pipeline=llm_pipe)

prompt = PromptTemplate(
    template="""Answer using ONLY the context below.
Be concise: 1–3 sentences for simple answers, up to 8 for complex ones.
Mention sources if possible. If unknown, reply: I don't know.

Context: {context}
Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)
print("✅ RetrievalQA ready")

# -----------------------------
# Ask a question
# -----------------------------
query = "what is LLMs"
print(f"\n❓ {query}")
ans = qa.invoke(query)

# -----------------------------
# Clean answer
# -----------------------------
def clean(ans) -> Tuple[str, List[str]]:
    txt = ans.get("result", "").strip()
    sources = [d.metadata.get("source_file") for d in ans.get("source_documents", [])]

    if not txt:
        return "I don't know", sources

    sentences = re.split(r'(?<=[.!?]) +', txt.replace("\n", " "))
    seen, result = set(), []
    for s in sentences:
        if s and s[-1] in ".!?":
            if s not in seen:
                result.append(s)
                seen.add(s)
        if len(result) >= 8:
            break
    return " ".join(result), sources

answer, sources = clean(ans)
print("\n📌 Answer:\n", answer)
if sources:
    print("\n📂 Sources:", ", ".join(set(sources)))
