# Architecture

The RAG Chatbot project consists of two implementations with similar architectures but different underlying frameworks:

## Common Architecture

1. **Document Ingestion**:
   - PDF documents are stored in the `data/pdfs/` directory.
   - These documents are processed and indexed for retrieval.

2. **Retrieval-Augmented Generation**:
   - A retrieval mechanism fetches relevant document snippets based on user queries.
   - A pre-trained language model generates responses using the retrieved snippets as context.

3. **Chat Interface**:
   - Provides an interface for users to interact with the chatbot.

## Huggingface Implementation

- **Framework**: Huggingface Transformers
- **Components**:
  - `RAG_Huggingface.py`: Core logic for retrieval and generation.
  - `Hf_chatbot.py`: Chat interface for user interaction.

## OLLAMA Implementation

- **Framework**: OLLAMA APIs
- **Components**:
  - `RAG_Ollama_train.py`: Handles training and indexing of documents.
  - `OLLAMA_chatbot.py`: Core logic for retrieval and generation.
  - `chat_with_VAD.py`: Adds voice activity detection for audio-based interaction.
  - `Chat_with_record.py`: Enables audio recording for user queries.