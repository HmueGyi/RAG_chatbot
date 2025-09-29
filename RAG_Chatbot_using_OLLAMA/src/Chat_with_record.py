import gradio as gr
import whisper
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
# from gtts import gTTS
from TTS.api import TTS
import tempfile
import base64

# -----------------------------
# Whisper STT Model (CPU)
# -----------------------------
stt_model = whisper.load_model("medium", device="cpu")
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)

def transcribe_audio(audio_path):
    """
    Convert audio file to text using Whisper (Myanmar language).
    """
    if audio_path is None:
        return ""
    result = stt_model.transcribe(
        audio_path,
        language="my",
        beam_size=5,
        temperature=0,
        fp16=False
    )

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
    raise RuntimeError(
        f"❌ No Chroma DB found at {CHROMA_DB_DIR}. Please create one first."
    )


# -----------------------------
# Retriever and Prompt
# -----------------------------
retriever = chroma_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

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
# Query function
# -----------------------------
UNKNOWN_RESPONSE = "I don't know how to response ."

def ask_question(query: str):
    """
    Ask a question to the RetrievalQA chain and format the response.
    """
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
            sources = [
                doc.metadata.get("source_file")
                for doc in source_docs
                if doc.metadata.get("source_file")
            ]
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
        """
        Respond to user input, update chat history, and generate TTS audio.
        """
        answer = ask_question(user_msg)
        chat_history.append((user_msg, answer))

        # Convert Myanmar names to phonetic (optional, if needed)
        tts_text = answer

        # TTS with Coqui TTS
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tts_model.tts_to_file(text=tts_text, file_path=tmpfile.name)
            audio_path = tmpfile.name

        # Convert WAV to base64 for autoplay
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        audio_player = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
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

