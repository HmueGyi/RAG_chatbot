# Usage

This guide explains how to use the RAG Chatbot implementations.

## Huggingface Implementation

1. **Navigate to the Source Directory**:
   ```bash
   cd RAG_Chatbot_using_Huggingface/src
   ```

2. **Run the Chatbot**:
   ```bash
   python Hf_chatbot.py
   ```

3. **Interact with the Chatbot**:
   - Enter your queries in the terminal.
   - The chatbot will respond based on the indexed documents.

## OLLAMA Implementation

1. **Navigate to the Source Directory**:
   ```bash
   cd RAG_Chatbot_using_OLLAMA/src
   ```

2. **Run the Chatbot**:
   - For text-based interaction:
     ```bash
     python OLLAMA_chatbot.py
     ```
   - For audio-based interaction:
     ```bash
     python chat_with_VAD.py
     ```

3. **Interact with the Chatbot**:
   - Enter your queries or speak into the microphone (for audio-based interaction).
   - The chatbot will respond based on the indexed documents.