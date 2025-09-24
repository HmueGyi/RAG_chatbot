from pathlib import Path
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# -----------------------------
# Embeddings and LLM
# -----------------------------
embed_model = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0.2, max_tokens=400)

# -----------------------------
# Load Chroma DB (no PDF/TXT loading)
# -----------------------------
if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
    print("✅ Loading existing Chroma DB...")
    chroma_db = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        collection_name="doc_collection",
        embedding_function=embed_model,
    )
else:
    raise RuntimeError(f"❌ No Chroma DB found at {CHROMA_DB_DIR}. Please create one first.")

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
# Query function
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
    if sources:
        paragraph += f"\n\n📂 Sources: {', '.join(set(sources))}"
    return paragraph

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📚 High-Accuracy RAG Chatbot (Chroma + Ollama)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question")

    def respond(user_msg, chat_history):
        answer = ask_question(user_msg)
        chat_history.append((user_msg, answer))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
