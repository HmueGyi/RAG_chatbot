import gradio as gr
import whisper
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from gtts import gTTS
import tempfile
import base64

# -----------------------------
# Whisper STT Model (CPU)
# -----------------------------
stt_model = whisper.load_model("medium", device="cpu")

def transcribe_audio(audio_path):
    """Convert audio file to text using Whisper"""
    if audio_path is None:
        return ""
    result = stt_model.transcribe(audio_path, beam_size=1, temperature=0, fp16=False)
    return result["text"]

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
retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

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

    if ans_text == UNKNOWN_RESPONSE or not ans_text:
        paragraph = UNKNOWN_RESPONSE
    else:
        paragraph = " ".join(ans_text.split())
        if source_docs:
            sources = [doc.metadata.get("source_file") for doc in source_docs if doc.metadata.get("source_file")]
            if sources:
                paragraph += f"\n\n📂 Sources: {', '.join(set(sources))}"

    return paragraph

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🎤 Voice-Enabled RAG Chatbot (STT + Chroma + Ollama + TTS)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your question or use microphone below")
    mic = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak Now")
    audio_output = gr.HTML()

    def respond(user_msg, chat_history):
        answer = ask_question(user_msg)
        chat_history.append((user_msg, answer))

        # TTS with gTTS
        tts = gTTS(text=answer, lang='en')
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

    # Text input
    msg.submit(respond, [msg, chatbot], [msg, chatbot, audio_output])

    # Mic input → Transcribe → Feed to chatbot
    def voice_to_chat(audio, chat_history):
        user_msg = transcribe_audio(audio)
        return respond(user_msg, chat_history)

    mic.stop_recording(voice_to_chat, [mic, chatbot], [msg, chatbot, audio_output])

demo.launch()
