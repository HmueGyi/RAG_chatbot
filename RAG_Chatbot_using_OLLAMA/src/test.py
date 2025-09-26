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
    """
    Convert audio file to text using Whisper (Myanmar language).
    """
    if audio_path is None:
        return ""
    result = stt_model.transcribe(
        audio_path,
        language="my",
        beam_size=1,
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
You are **SiSi**, a friendly and caring AI assistant.  

You handle three types of conversations:

1. **Daily conversations**:  
   - Respond naturally and warmly.  
   - Keep replies short (1–2 sentences), direct, and human-like.  
   - Stay consistent: you are not ChatGPT, Ollama, or Llama — only SiSi.  
   - Never use filler phrases like "I'd be happy to help you" or "Sure, here you go."  

2. **Knowledge/document questions**:  
   - Use ONLY the given context.  
   - Answer concisely and factually (1–3 sentences).  
   - If the answer is not in the context → reply politely like:  
     "It seems like '<user question>' might be a person or topic, but I couldn't find any information in the context. If you could provide more details, I'd be happy to try to help further."  
   - Do not invent information.  

3. **Personal information about people**:  
   - If asked about a person (name, position, experiences, investments), only use the context.  
   - Be concise:  
       - Position → just the position.  
       - Investments → just the investments.  
       - If nothing found → use the polite unknown message above.  

Rules:   
- Always stay friendly and helpful.  
- Never invent details or use prior knowledge outside the context.  

Context:
{context}

Question:
{question}

Answer as SiSi:
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
# Myanmar Name Phonetic Mapping
# -----------------------------
MYANMAR_TO_PHONETIC = {
    "စိုး": "Soe", "မောင်": "Maung", "နောင်": "Naung", "ကောင်း " : "Kaung" ,"ထက် " : "Htet",
    "ခိုင်": "Khine", "နေ": "Nay", "ထွန်း": "Htun", "ခန့်": "Khant",
    "ဇော်": "Zaw", "စံ": "San", "မြင့်": "Myint", "ကျော်": "Kyaw",
    "မင်း": "Min", "စန္ဒာ": "Sandar", "သန်း": "Than", "ဇေယျာ": "Zayar",
    "အောင်": "Aung", "သိန်း": "Thein", "နီ": "Ni", "ညို": "Nyo",
    "စိုးမောင်": "Soe Maung", "စိုးနောင်": "Soe Naung", "ပြည့်": "Pyae",
    "ဦး": "U", "ဒေါ်": "Daw", "ဖျိုး": "Phyo", "သော်န်": "Thant",
    "ခန့်": "Khant", "ဇင်": "Zin", "မေ": "Me", "လွင်": "Lwin",
    "ဆွာ": "Swar", "သု": "Thu", "မြတ်": "Myat", "ခန့်": "Khant",
    "မောင်မောင်": "Maung Maung", "ဝင်း": "Win", "စုံ": "Soan" , "စုံ": "Sone"
}

def convert_myanmar_to_phonetic(text):
    """
    Replace Myanmar names in the text with phonetic equivalents.
    """
    for my_name, phonetic in MYANMAR_TO_PHONETIC.items():
        text = text.replace(my_name, phonetic)
    return text


# -----------------------------
# Gradio UI
# -----------------------------
# ---------------- Placeholder functions ---------------- #
def ask_question(user_msg):
    return f"Echo: {user_msg}"  # Replace with your RAG/Ollama logic

def convert_myanmar_to_phonetic(text):
    return text  # Replace with your phonetic conversion logic

def transcribe_audio(audio_path):
    return "Transcribed text from audio"  # Replace with your STT logic

# ---------------- Gradio Interface ---------------- #
with gr.Blocks() as demo:
    gr.Markdown("## 🎤 Voice-Enabled ChatGPT-Style RAG Chatbot")

    # Chat display
    chatbot = gr.Chatbot(elem_id="chatbot", label="Chat", type="messages")  # Use messages type

    # User input row: text + mic
    with gr.Row():
        msg = gr.Textbox(
            label="Type your question",
            placeholder="Ask me anything...",
            lines=1
        )
        mic = gr.Audio(
            label="🎤 Speak Now",
            type="filepath",  # record to file
            streaming=False
        )

    # Audio output for TTS
    audio_output = gr.HTML()

    # ---------------- Functions ---------------- #
    def respond(user_msg, chat_history):
        """
        Handle user text, update chat, and generate TTS.
        """
        answer = ask_question(user_msg)

        # ChatGPT-style message appending
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": answer})

        # TTS conversion
        tts_text = convert_myanmar_to_phonetic(answer)
        tts = gTTS(text=tts_text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts.save(tmpfile.name)
            audio_path = tmpfile.name

        # Convert MP3 to base64 for HTML audio
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        audio_player = f"""
        <audio autoplay controls>
            <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">
        </audio>
        """

        return "", chat_history, audio_player

    def voice_to_chat(audio_file, chat_history):
        """
        Transcribe audio and feed it to respond function.
        """
        if audio_file is None:
            return "", chat_history, ""
        user_msg = transcribe_audio(audio_file)
        return respond(user_msg, chat_history)

    # ---------------- Event Bindings ---------------- #
    msg.submit(respond, [msg, chatbot], [msg, chatbot, audio_output])
    mic.change(voice_to_chat, [mic, chatbot], [msg, chatbot, audio_output])

demo.launch()