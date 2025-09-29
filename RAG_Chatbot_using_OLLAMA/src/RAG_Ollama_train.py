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
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", ". ", "\n"]
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
retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

qa_prompt = PromptTemplate(
    template="""
You are **CiCi**, a friendly and caring AI assistant.  

You handle three types of conversations:

1. **Daily conversations**:  
   - Respond naturally and warmly.  
   - Keep replies short (1–2 sentences), direct, and human-like.  
   - Stay consistent: you are not ChatGPT, Ollama, or Llama — only CiCi.
   - Never use filler phrases like "I'd be happy to help you" or "Sure, here you go."  

2. **Knowledge/document questions**:  
   - Use ONLY the given context.  
   - If the question is a **wh-question** (who, what, when, where, why, how), provide a **detailed answer in 3–5 sentences**, explaining clearly and providing context from the documents.  
   - For non-wh questions, answer concisely and factually (1–2 sentences).  
   - If the answer is not in the context → reply politely like:  
     "It seems like '<user question>' might be a person or topic, but I couldn't find any information in the context. If you could provide more details, I'd be happy to try to help further."  
   - Do not invent information.  

3. **Personal information about people**:  
   - If asked about a person (name, position, experiences, investments), only use the context.  
   - Be concise:  
       - Position → just the position.  
       - Investments → just the investments.  
       - If nothing found → use the polite unknown message above.  

**General Rules:**   
- Always stay friendly, clear, and helpful.  
- Never invent details or use prior knowledge outside the context.  
- Adjust answer length according to question type:
    - Wh-questions → 3–5 sentences, detailed.
    - Other questions → 1–2 sentences, concise.  

Context:
{context}

Question:
{question}

Answer as CiCi:
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
        # print("📂 Sources:", ", ".join(sources))

# -----------------------------
# Example Usage
# -----------------------------
ask_question("who are you")
