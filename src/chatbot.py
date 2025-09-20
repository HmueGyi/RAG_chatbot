import gradio as gr
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# -----------------------------
# Load embeddings
# -----------------------------
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # change to "cuda" for GPU
    encode_kwargs={"normalize_embeddings": False},
    cache_folder="./models",
    multi_process=False,
)

# -----------------------------
# Load Chroma database
# -----------------------------
chroma_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embed_model
)

# -----------------------------
# Setup retriever
# -----------------------------
retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8},
)

# -----------------------------
# Load TinyLlama model
# -----------------------------
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)

tinyllama_pipeline = pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",   # "cuda" if GPU, "cpu" otherwise
        torch_dtype="auto",
        cache_dir="./models",
    ),
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)

tinyllama_pipeline_hf = HuggingFacePipeline(pipeline=tinyllama_pipeline)

# -----------------------------
# Setup RetrievalQA
# -----------------------------
retrievalQA = RetrievalQA.from_llm(
    llm=tinyllama_pipeline_hf,
    retriever=retriever
)

# -----------------------------
# Chatbot function
# -----------------------------
def chatbot(input_text: str) -> str:
    ans = retrievalQA.invoke(input=input_text)
    return ans["result"]

# -----------------------------
# Gradio interface
# -----------------------------
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=7, label="Enter your text"),
    outputs="text",
    title="Information Retrieval Bot (TinyLlama)",
)

iface.launch(share=True)
