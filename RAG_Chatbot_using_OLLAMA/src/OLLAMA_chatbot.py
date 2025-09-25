from pathlib import Path
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from gtts import gTTS
import tempfile
import base64

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
# Load Chroma DB
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
- If the answer cannot be found in the context, respond exactly with: I don't know how to response .
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
UNKNOWN_RESPONSE = "I don't know how to response ."

def ask_question(query: str):
    ans = retrievalQA.invoke(query)
    if isinstance(ans, dict) and "result" in ans:
        ans_text = ans["result"].strip()
        source_docs = ans.get("source_documents", [])
    else:
        ans_text = str(ans).strip()
        source_docs = []

    # Standardize unknown response
    if ans_text == UNKNOWN_RESPONSE or not ans_text:
        paragraph = UNKNOWN_RESPONSE
    else:
        paragraph = " ".join(ans_text.split())
        # Only add sources if not the unknown response
        if source_docs:
            sources = [doc.metadata.get("source_file") for doc in source_docs if doc.metadata.get("source_file")]
            if sources:
                paragraph += f"\n\n📂 Sources: {', '.join(set(sources))}"

    return paragraph

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📚 High-Accuracy RAG Chatbot (Chroma + Ollama + 🎤 TTS Auto-Play)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question")
    audio_output = gr.HTML()

    def respond(user_msg, chat_history):
        answer = ask_question(user_msg)

        # TTS text matches the answer exactly
        tts_text = answer

        chat_history.append((user_msg, answer))

        # Generate TTS using gTTS
        tts = gTTS(text=tts_text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts.save(tmpfile.name)
            audio_path = tmpfile.name

        # Convert MP3 to base64 for autoplay
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        audio_player = f"""
        <audio autoplay>
            <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">
        </audio>
        """

        return "", chat_history, audio_player

    msg.submit(respond, [msg, chatbot], [msg, chatbot, audio_output])

demo.launch()
