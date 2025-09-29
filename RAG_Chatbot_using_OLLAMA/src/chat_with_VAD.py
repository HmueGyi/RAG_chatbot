import gradio as gr
import whisper
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from TTS.api import TTS
import tempfile
import base64

# -----------------------------
# Models
# -----------------------------
stt_model = whisper.load_model("medium", device="cpu")
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)

embed_model = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0.2, max_tokens=400)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

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


retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -----------------------------
# Prompt Template
# -----------------------------
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
# Functions
# -----------------------------
def transcribe_audio(audio_path):
    """Convert audio file to text using Whisper."""
    if audio_path is None:
        return ""
    result = stt_model.transcribe(audio_path, language="my", beam_size=5, temperature=0, fp16=False)
    return result["text"]

def ask_question(query: str):
    """Query the RAG pipeline and return response with optional sources."""
    ans = retrievalQA.invoke(query)
    if isinstance(ans, dict) and "result" in ans:
        ans_text = ans["result"].strip()
        source_docs = ans.get("source_documents", [])
    else:
        ans_text = str(ans).strip()
        source_docs = []

    if not ans_text:
        return "I don't know how to respond."

    paragraph = " ".join(ans_text.split())
    if source_docs:
        sources = [doc.metadata.get("source_file") for doc in source_docs if doc.metadata.get("source_file")]
        if sources:
            paragraph += f"\n\n📂 Sources: {', '.join(set(sources))}"

    return paragraph

def respond(user_msg, chat_history):
    """Generate response and TTS audio."""
    answer = ask_question(user_msg)
    chat_history.append((user_msg, answer))

    # TTS to WAV → base64 for HTML audio player
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tts_model.tts_to_file(text=answer, file_path=tmpfile.name)

    with open(tmpfile.name, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    audio_player = f"""
    <audio autoplay>
        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
    </audio>
    """

    return "", chat_history, audio_player

def voice_to_chat(audio, chat_history):
    """Transcribe microphone input and feed to chatbot."""
    user_msg = transcribe_audio(audio)
    chat_msg, chat_history, audio_player = respond(user_msg, chat_history)
    
    # Clear the microphone input after each recording
    return None, chat_history, audio_player

# -----------------------------
# VAD JS snippet
# -----------------------------
js_vad = """
async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1);

  const script2 = document.createElement("script");
  script2.onload = async () =>  {
    const record = document.querySelector('.record-button');
    if (record) { record.textContent = "Just Start Talking!"; record.style.width = "fit-content"; }

    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        const record = document.querySelector('.record-button');
        const player = document.querySelector('#streaming-out');
        if (record && (player == null || player.paused)) { record.click(); }
      },
      onSpeechEnd: (audio) => {
        const stop = document.querySelector('.stop-button');
        if (stop) { stop.click(); }
      }
    });
    myvad.start();
  };

  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js";
  script1.onload = () => document.head.appendChild(script2);
}
"""

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(js=js_vad) as demo:
    gr.Markdown("## 🎤 Voice-Enabled RAG Chatbot (Whisper + Chroma + Ollama + TTS + VAD)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your question or use microphone below")
    mic = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak Now")
    audio_output = gr.HTML()

    msg.submit(respond, [msg, chatbot], [msg, chatbot, audio_output])
    mic.stop_recording(voice_to_chat, [mic, chatbot], [mic, chatbot, audio_output])


demo.launch()