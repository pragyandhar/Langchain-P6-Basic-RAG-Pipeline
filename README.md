# 🚀 Langchain P6: Basic RAG Pipeline

A complete **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain that empowers you to ask questions about your documents and get answers grounded strictly in the provided context.

---

## ✨ Overview

This project brings together all essential RAG components—document loading, text chunking, vector embeddings, semantic search, and LLM-powered answer generation—into a seamless, interactive pipeline. It demonstrates how to build a production-ready system that prevents hallucinations by grounding responses exclusively in your documents.

---

## 🎯 Key Features

- **📄 Multi-Format Support** – Load and process both PDF and TXT documents
- **🔗 Smart Text Chunking** – RecursiveCharacterTextSplitter with configurable overlap for context preservation
- **🧠 Vector Embeddings** – OpenAI embeddings with FAISS vector store for lightning-fast semantic search
- **🎯 Relevance Scoring** – See similarity scores for retrieved chunks before generating answers
- **🔐 Context-Grounded Answers** – LLM strictly follows retrieved context; no hallucinations
- **📍 Source Tracking** – Automatic citation of sources used in each answer
- **⚡ Interactive Query Interface** – Real-time Q&A with command support

---

## 🧩 Architecture

The pipeline flows through these key components:

```
User Input
    ↓
Document Loading (loader.py)
    ↓
Text Splitting (splitter.py)
    ↓
Embeddings & Vector Store (vectorstore.py)
    ↓
Semantic Retrieval
    ↓
RAG Chain (chain.py) + LLM
    ↓
Grounded Answer + Sources
```

### Core Modules

| Module | Purpose |
|--------|---------|
| **main.py** | Orchestrates the entire pipeline and interactive loop |
| **loader.py** | Loads PDF and TXT documents |
| **splitter.py** | Chunks documents intelligently with overlap |
| **vectorstore.py** | Builds FAISS vector store and retriever |
| **chain.py** | Constructs RAG chain with LLM instructions |

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository:**
   ```bash
   cd Langchain-P6-Basic-RAG-Pipeline
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

---

## 🚀 Usage

### Running the Pipeline

```bash
python main.py
```

### Interactive Workflow

1. **Load Documents** – Enter file paths when prompted (supports multiple files)
   ```
   File path: document1.pdf
   File path: document2.txt
   File path: <press Enter to finish>
   ```

2. **Ask Questions** – Query your documents naturally
   ```
   Your question: What is mentioned about climate change?
   ```

3. **View Relevance** – See the top relevant chunks with similarity scores:
   ```
   Top 3 relevant chunks for query: '...'
   Score: 0.8234 | Chunk: Climate change refers to...
   Score: 0.7891 | Chunk: Global warming impacts...
   ```

4. **Get Grounded Answers** – Receive answers strictly based on your documents:
   ```
   Answer: According to the documents, climate change is... [source: document1.pdf]
   ```

5. **Track Sources** – View all sources used:
   ```
   [Sources used]
   - document1.pdf (page 1)
   - document2.txt
   ```

### Example Commands

| Command | Action |
|---------|--------|
| Ask any question | Get a context-grounded answer |
| `sources` | List all loaded documents |
| `quit` or `exit` | Exit the application |

---

## ⚙️ Configuration

Customize behavior in the relevant modules:

### Document Splitting (`splitter.py`)
```python
chunks = splitter(documents, chunk_size=1000, chunk_overlap=200)
```
- `chunk_size`: Size of each text chunk (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks to preserve context (default: 200 characters)

### Retrieval (`vectorstore.py`)
```python
retriever = build_retriever(vectorstore, k=3)
```
- `k`: Number of top chunks to retrieve per query (default: 3)

### LLM Model (`chain.py`)
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```
- `model`: Change to any OpenAI model (e.g., "gpt-4")
- `temperature`: Lower = more deterministic, higher = more creative

---

## 📊 How It Works

1. **Loading** – Documents are loaded using appropriate loaders (PDF/TXT)
2. **Chunking** – Text is split into overlapping chunks using RecursiveCharacterTextSplitter
3. **Embedding** – Each chunk is converted to a dense vector using OpenAI embeddings
4. **Storage** – Vectors are indexed in FAISS for fast similarity search
5. **Retrieval** – User query is embedded and matched against stored vectors
6. **Generation** – Retrieved context is passed to GPT-4o-mini with strict instructions
7. **Response** – LLM generates answers grounded exclusively in the retrieved context

---

## 🔒 Safety Features

The RAG chain implements strict guardrails:

- **Context-Only Responses** – System prompt forbids knowledge outside provided documents
- **No Hallucinations** – LLM explicitly instructed to say "I don't know" if answer isn't in context
- **Source Attribution** – All answers must cite their sources
- **Temperature = 0** – Deterministic outputs for consistency

---

## 📦 Dependencies

- `langchain` – LLM framework
- `langchain-openai` – OpenAI integration
- `langchain-community` – Document loaders and vector stores
- `langchain-text-splitters` – Text splitting utilities
- `python-dotenv` – Environment variable management
- `faiss-cpu` – Vector similarity search

---

## 📝 Example

**Sample data files included:**
- `rag1.txt` – Example document 1
- `rag2.txt` – Example document 2

Try the pipeline with these files:
```bash
python main.py
# Enter: rag1.txt
# Enter: rag2.txt
# Ask: What topics are covered?
```

---

## 🤝 Next Steps & Extensions

- Switch to different embedding models (e.g., Hugging Face, Cohere)
- Implement metadata filtering for advanced retrieval
- Add document preprocessing (cleaning, normalization)
- Integrate with different LLMs (Claude, Llama, etc.)
- Add chat history for multi-turn conversations
- Deploy as a web API or chatbot interface

---

## ⚠️ Important Notes

- **API Costs** – Using OpenAI APIs incurs costs. Monitor your usage
- **Token Limits** – Large documents may exceed token limits; adjust chunk sizes accordingly
- **Environment Variables** – Keep your `.env` file secure and never commit it to version control

---

## 📄 License

This project is for educational purposes. Feel free to modify and extend it!

---

**Built with ❤️ using LangChain, OpenAI, and FAISS**
