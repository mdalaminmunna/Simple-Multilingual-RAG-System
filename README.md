# Setup Guidelines for Multilingual RAG Project

Below are clear setup instructions for two scenarios: (1) running the pipeline as a command-line script without an API, and (2) deploying it as a REST API using FastAPI.

## 1. Setup Without API (Command-Line Only)

### Prerequisites

- Python 3.8 or higher
- Download the required PDF file (e.g., `HSC26_Bangla_1st_paper.pdf`) and place it in your project directory.

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/simple-multilingual-rag.git
   cd simple-multilingual-rag
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Weights**
   - The needed transformer model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) will be downloaded automatically on first run.

4. **Place PDF File**
   - Ensure `HSC26_Bangla_1st_paper.pdf` is in the project directory.

5. **Run the Script**
   ```bash
   python simple_multilingual_rag.py
   ```
   - You will see a prompt to type your question (in English or Bengali).
   - Example: `অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?`

## 2. Setup With API Implementation (FastAPI)

### Prerequisites

- Python 3.8 or higher
- Download the PDF file (`HSC26_Bangla_1st_paper.pdf`) as above.

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/simple-multilingual-rag.git
   cd simple-multilingual-rag
   ```

2. **Install All Dependencies (Including API)**
   ```bash
   pip install -r requirements.txt
   pip install fastapi uvicorn
   ```

3. **Place PDF File**
   - Copy `HSC26_Bangla_1st_paper.pdf` into your project directory.

4. **Run the API Server**
   ```bash
   uvicorn api_rag:app --reload
   ```
   - The API server will be available at `http://127.0.0.1:8000`
   - Interactive API docs: `http://127.0.0.1:8000/docs`

5. **Example API Usage**
   - Send a POST request to `/ask` with a JSON body:  
     ```json
     {
       "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
     }
     ```
   - You will receive a response with the retrieved answer.

**Tip:**  
- You may customize chunk size and relevant parameters inside the script before running.
- After setup, refer to project documentation for additional configuration options or evaluation methods.
