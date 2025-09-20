"""
This script demonstrates how to create vector embeddings
on custom data (PDFs) and build a Retrieval-based chatbot.

Process: Load -> Split -> Store -> Retrieve -> Generate
References:
- https://python.langchain.com/docs/use_cases/question_answering/
- https://python.langchain.com/docs/modules/chains/#legacy-chains
"""

from pathlib import Path
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, pipeline
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# Define PDF file path
# -----------------------------
pdf_file_dir_path = str(
    Path(__file__).resolve().parent.parent.parent.joinpath(r"./Book_Chatbot/data", "pdfs")
)
print(f"PDF Directory Path: {pdf_file_dir_path}")

# -----------------------------
# Split documents into smaller chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# -----------------------------
# Load data from PDFs using PyPDFLoader
# -----------------------------
pdf_dir = Path(pdf_file_dir_path)
loaders = [PyPDFLoader(str(pdf)) for pdf in pdf_dir.glob("*.pdf")]
pages = []
for l in loaders:
    pages.extend(l.load_and_split(text_splitter=text_splitter))

# -----------------------------
# Generate embeddings
# -----------------------------
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=r"./Book/sentence_transformers",
    model_kwargs={"device": "cpu"},  # set to "cuda" if GPU is available
    encode_kwargs={"normalize_embeddings": False},
    multi_process=False,
)

# -----------------------------
# Store embeddings in Chroma
# -----------------------------
chroma_db = Chroma.from_documents(
    pages, embed_model, persist_directory="./chroma_db"
)

# -----------------------------
# Setup retriever
# -----------------------------
retriever = chroma_db.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": 8},  # max relevant docs to retrieve
)

# -----------------------------
# Load Dolly (TinyLlama) model for Q&A
# -----------------------------
dolly_generate_text = pipeline(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    trust_remote_code=True,
    device_map="auto",  # -1 for CPU, 0 for GPU
    return_full_text=True,
    tokenizer=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    temperature=0.1,
    max_new_tokens=1000,
    model_kwargs={
        "cache_dir": "./models",
        "offload_folder": "offload",  # use for large models >7B
    },
)

dolly_pipeline_hf = HuggingFacePipeline(pipeline=dolly_generate_text)

# -----------------------------
# Setup prompt and output parser
# -----------------------------
question_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Question: {question}
"""
prompt_template = ChatPromptTemplate.from_template(question_template)
output_parser = StrOutputParser()
chain_1 = prompt_template | dolly_pipeline_hf | output_parser

# Example: Direct question to Dolly model
chain_1_ans = chain_1.invoke(
    input={"question": "Provide NVIDIA’s outlook for the third quarter of fiscal 2024"}
)
print("Chain 1 Answer:\n", chain_1_ans)

# -----------------------------
# Create RetrievalQA for custom data
# -----------------------------
retrievalQA = RetrievalQA.from_llm(
    llm=dolly_pipeline_hf,
    retriever=retriever
)
print("RetrievalQA Setup:\n", retrievalQA)

# Ask question using custom PDF data
ans = retrievalQA.invoke(
    "Provide NVIDIA’s outlook for the third quarter of fiscal 2024"
)
print("RetrievalQA Answer:\n", ans)
