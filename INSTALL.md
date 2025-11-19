# Installation

Follow these steps to set up the RAG Chatbot project:

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (optional but recommended)

## Steps

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd RAG_chatbot
   ```

2. **Set Up Virtual Environment** (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   - For Huggingface implementation:
     ```bash
     cd RAG_Chatbot_using_Huggingface/src
     pip install -r requirements.txt
     ```
   - For OLLAMA implementation:
     ```bash
     cd RAG_Chatbot_using_OLLAMA/src
     pip install -r requirements.txt
     ```

4. **Verify Installation**:
   - Run a test script to ensure everything is set up correctly.