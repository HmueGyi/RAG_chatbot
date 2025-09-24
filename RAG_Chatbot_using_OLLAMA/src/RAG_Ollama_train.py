"""
High-accuracy RAG Chatbot with Ollama:
Load PDFs, text files, images, Word docs,
create embeddings, store in Chroma DB,
and query with Llama3.1 via Ollama.
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

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIRS = {
    "pdf": BASE_DIR / "data" / "pdfs",
    "txt": BASE_DIR / "data" / "txts",
    "img": BASE_DIR / "data" / "images",
    "docx": BASE_DIR / "data" / "docs",
}
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# -----------------------------
# Text Splitter
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    length_function=len,
)

# -----------------------------
# Embeddings and LLM
# -----------------------------
embed_model = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0.2, max_tokens=400)

# -----------------------------
# Load or Create Chroma DB
# -----------------------------
def load_documents():
    docs = []
    for dtype, path in DATA_DIRS.items():
        for file in path.glob("*.*"):
            try:
                if dtype == "pdf":
                    loader = PyPDFLoader(str(file))
                elif dtype == "txt":
                    loader = TextLoader(str(file), encoding="utf-8")
                elif dtype == "img":
                    loader = UnstructuredImageLoader(str(file))
                elif dtype == "docx":
                    loader = UnstructuredWordDocumentLoader(str(file))
                else:
                    continue

                file_docs = loader.load_and_split(text_splitter=text_splitter)
                for d in file_docs:
                    d.metadata["source_file"] = file.name
                docs.extend(file_docs)
            except Exception as e:
                print(f"⚠️ Failed to load {dtype} {file.name}: {e}")
    return docs

if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
    print("Loading existing Chroma DB...")
    chroma_db = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        collection_name="doc_collection",
        embedding_function=embed_model,
    )
else:
    print("Creating new Chroma DB from documents...")
    pages = load_documents()
    print(f"Total chunks before deduplication: {len(pages)}")

    # Deduplicate chunks
    seen = set()
    unique_pages = []
    for p in pages:
        content_hash = hash(p.page_content)
        if content_hash not in seen:
            unique_pages.append(p)
            seen.add(content_hash)

    print(f"Total unique chunks after deduplication: {len(unique_pages)}")

    chroma_db = Chroma.from_documents(
        documents=unique_pages,
        embedding=embed_model,
        persist_directory=str(CHROMA_DB_DIR),
        collection_name="doc_collection",
    )
    print(f"Chroma DB created at {CHROMA_DB_DIR}")

# -----------------------------
# Retriever and Prompt
# -----------------------------
retriever = chroma_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

qa_prompt = PromptTemplate(
    template="""
You are a helpful assistant. Answer the question using ONLY the information provided in the context below.

Guidelines:
- Provide a coherent paragraph as the answer.
- Always cite the source filename for each fact if available.
- If the answer cannot be found in the context, respond exactly with: I don't know.
- Do NOT use any outside knowledge.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
)

# -----------------------------
# Query and Post-process
# -----------------------------
def ask_question(query: str):
    ans = retrievalQA.invoke(query)
    if isinstance(ans, dict) and "result" in ans:
        ans_text = ans["result"].strip()
        sources = [doc.metadata.get("source_file") for doc in ans.get("source_documents", [])]
    else:
        ans_text = str(ans).strip()
        sources = []

    if not ans_text or ans_text.lower() in ["", "none", "unknown"]:
        ans_text = "I don't know"

    paragraph = " ".join(ans_text.split())
    print("\n📌 Answer:\n", paragraph)
    if sources:
        print("\n📂 Sources:", ", ".join(set(sources)))

# Example usage
ask_question("how much weight can kilobot carry")

