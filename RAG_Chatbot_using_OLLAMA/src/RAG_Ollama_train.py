"""
High-accuracy RAG Chatbot with Ollama:
Load PDFs, text files, images, Word docs,
create embeddings, store in Chroma DB,
and query with Llama2 via Ollama.
Concise and exact answers without sources.
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
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". "]
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
# Use similarity search with a slightly higher k for more coverage
retriever = chroma_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}  # increased k to improve answer accuracy
)

qa_prompt = PromptTemplate(
    template="""
You are an expert assistant specialized in providing **accurate, concise, and fact-based answers**. 
Answer the user's question using ONLY the information provided in the context. Do not use any prior knowledge or assumptions.

Instructions:
1. Read all context carefully and synthesize a single coherent paragraph as your answer.
2. Always cite the source filename for each fact or statement if available.
3. If the answer is not present in the context, respond exactly with: I don't know.
4. Avoid adding any explanations, opinions, or content not supported by the context.
5. If multiple context chunks contain conflicting information, prioritize the majority or clearly indicate uncertainty.

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
# Query Function with Verification
# -----------------------------
def ask_question(query: str):
    result = retrievalQA.invoke(query)
    
    # Extract answer text
    if isinstance(result, dict) and "result" in result:
        answer_text = result["result"].strip()
    else:
        answer_text = str(result).strip()
    
    # Safety fallback
    if not answer_text or answer_text.lower() in ["", "none", "unknown"]:
        answer_text = "I don't know"
    
    # Clean formatting
    answer_text = " ".join(answer_text.split())
    
    print("\n📌 Answer:\n", answer_text)
    
    # Optional: print sources
    if isinstance(result, dict) and "source_documents" in result:
        sources = {doc.metadata.get("source", "unknown") for doc in result["source_documents"]}
        print("📂 Sources:", ", ".join(sources))

# -----------------------------
# Example Usage
# -----------------------------
ask_question("Who is U VanLar Lura")
