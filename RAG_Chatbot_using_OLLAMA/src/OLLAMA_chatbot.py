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
