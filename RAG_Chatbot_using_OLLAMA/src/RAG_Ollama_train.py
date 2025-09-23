"""
High-accuracy RAG Chatbot:
Load PDFs, text files, images, Word docs,
create embeddings, store in Chroma DB,
and query with TinyLlama (via Ollama).
Source-aware and improved accuracy.
"""
"""
This script demonstrates how to create vector embeddings
on custom PDF data and build a Retrieval-based chatbot.

Process: Load -> Split -> Deduplicate -> Store -> Retrieve -> Generate
"""

"""
High-accuracy RAG Chatbot with Ollama:
Load PDFs, text files, images, Word docs,
create embeddings, store in Chroma DB,
and query with Llama2 via Ollama.
Source-aware and improved accuracy.
"""

from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredImageLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.docstore.document import Document
import re

# -----------------------------
# Define base paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
TXT_DIR = BASE_DIR / "data" / "txts"
IMG_DIR = BASE_DIR / "data" / "images"
DOCX_DIR = BASE_DIR / "data" / "docs"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

print(f"Data directories: PDFs={PDF_DIR}, Texts={TXT_DIR}, Images={IMG_DIR}, Docs={DOCX_DIR}")

# -----------------------------
# Split documents into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,
    length_function=len,
    is_separator_regex=False,
)

# -----------------------------
# Ollama embeddings (CPU-friendly)
# -----------------------------
embed_model = OllamaEmbeddings(model="llama2")

# -----------------------------
# Load or create Chroma DB
# -----------------------------
if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
    print("Loading existing Chroma DB...")
    chroma_db = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        collection_name="document_collection",
        embedding_function=embed_model,
    )
else:
    pages = []

    # --- PDFs ---
    for pdf_file in PDF_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load_and_split(text_splitter=text_splitter)
        for d in docs:
            d.metadata["source_file"] = pdf_file.name
        pages.extend(docs)

    # --- Text files ---
    for txt_file in TXT_DIR.glob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs = loader.load_and_split(text_splitter=text_splitter)
        for d in docs:
            d.metadata["source_file"] = txt_file.name
        pages.extend(docs)

    # --- Images ---
    for img_file in IMG_DIR.glob("*.*"):
        try:
            loader = UnstructuredImageLoader(str(img_file))
            docs = loader.load_and_split(text_splitter=text_splitter)
            for d in docs:
                d.metadata["source_file"] = img_file.name
            pages.extend(docs)
        except Exception as e:
            print(f"Failed to load image {img_file}: {e}")

    # --- Word documents ---
    for doc_file in DOCX_DIR.glob("*.docx"):
        loader = UnstructuredWordDocumentLoader(str(doc_file))
        docs = loader.load_and_split(text_splitter=text_splitter)
        for d in docs:
            d.metadata["source_file"] = doc_file.name
        pages.extend(docs)

    print(f"Total chunks created: {len(pages)}")

    # Deduplicate
    unique_pages = []
    seen_texts = set()
    for page in pages:
        if page.page_content not in seen_texts:
            unique_pages.append(page)
            seen_texts.add(page.page_content)
    pages = unique_pages
    print(f"Total unique chunks after deduplication: {len(pages)}")

    # Create Chroma DB
    chroma_db = Chroma.from_documents(
        documents=pages,
        embedding=embed_model,
        persist_directory=str(CHROMA_DB_DIR),
        collection_name="document_collection"
    )
    print(f"Chroma DB created at {CHROMA_DB_DIR}")

# -----------------------------
# Setup retriever
# -----------------------------
retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5},
)

# -----------------------------
# Setup Ollama chat model
# -----------------------------
llm = ChatOllama(model="llama2", temperature=0.2, max_tokens=600)

# -----------------------------
# Adaptive-length, source-aware prompt
# -----------------------------
qa_prompt = PromptTemplate(
    template="""
Answer the following question using ONLY the context provided.
Be concise: use 1–3 sentences for simple answers, up to 8 sentences for complex answers.
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
# Create RetrievalQA
# -----------------------------
retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
)

print("RetrievalQA ready ✅")

# -----------------------------
# Ask question
# -----------------------------
query = "where is rom dynamics located and what do they do?"
print(f"\n❓Question: {query}")

ans = retrievalQA.invoke(query)

# -----------------------------
# Post-process answer
# -----------------------------
if isinstance(ans, dict) and "result" in ans:
    ans_text = ans["result"].strip()
    sources = [doc.metadata.get("source_file") for doc in ans.get("source_documents", [])]
else:
    ans_text = str(ans).strip()
    sources = []

if not ans_text or ans_text.lower() in ["", "none", "unknown"]:
    ans_text = "I don't know"

# Keep up to 8 sentences
sentences = re.split(r'(?<=[.!?]) +', ans_text.replace("\n", " "))
seen = set()
filtered_ans = []
for s in sentences:
    s = s.strip()
    if s and s not in seen and s[-1] in ".!?":
        filtered_ans.append(s)
        seen.add(s)
    if len(filtered_ans) >= 8:
        break
ans_text = " ".join(filtered_ans) if filtered_ans else ans_text

print("\n📌RetrievalQA Answer:\n")
print(ans_text)
if sources:
    print("\n📂Sources:")
    print(", ".join(set(sources)))


