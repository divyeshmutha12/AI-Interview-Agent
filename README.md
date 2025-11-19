# AI Interviewer System 🎯

An intelligent AI-powered interview system that generates interview questions and evaluates candidate answers using multi-agent architecture.

## 🏗️ Architecture

Built with **LangChain Supervisor Pattern**:
- **Supervisor Agent**: Routes requests to specialized agents
- **Question Generator Agent**: Generates interview questions with ideal answers
  - Candidate Analysis Agent
  - KB Search Agent 
  - Web Search Agent 
- **Evaluation Pipeline**: 4-stage sequential evaluation system

## 🚀 Setup Instructions

### Prerequisites
- Python 3.13+
- OpenAI API Key
- Tavily API Key 
- Langfuse Account (optional, for observability)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/divyeshmutha12/AI-Interview-Agent.git
cd AI-Interview-Agent
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:
```
# OpenAI Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# OpenAI Embeddings for Knowledge Base (Required)
EMBEDDING_MODEL=text-embedding-3-small

# Langfuse Configuration (Optional)
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROMPT_LABEL=divyesh

# Tavily Web Search (Optional but Recommended)
TAVILY_API_KEY=your_tavily_api_key_here
```

5. **Get Tavily API Key (Optional but Recommended)**

The Web Search Agent uses Tavily API for real-time web searches:
- Visit [https://tavily.com](https://tavily.com)
- Sign up for a free account
- Get your API key (1000 searches/month free)
- Add it to `.env` as `TAVILY_API_KEY`


## 💻 Usage

### Web Interface (Recommended)
```bash
streamlit run web_app.py
```
Then open http://localhost:8501 in your browser.

### CLI Interface
```bash
python ai_interviewer/run_interview.py
```

## 📁 Project Structure

```
Supervisor_Agent/
├── ai_interviewer/
│   ├── agents/
│   │   ├── supervisor_agent.py      # Main supervisor
│   │   ├── sub_agents.py            # Sub-agents & evaluation pipeline
│   │   └── agent_service.py         # Service orchestration
│   ├── utils/
│   │   ├── document_parser.py       # PDF/DOCX/TXT parsing
│   │   └── knowledge_base_manager.py # FAISS KB manager
│   └── run_interview.py             # CLI interface
├── web_app.py                       # Streamlit web interface (with KB management)
├── knowledge_base/                  # FAISS vector database (auto-created)
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
└── README.md                        # This file
```

## 🔧 Tech Stack

- **LangChain** - Agent orchestration
- **LangGraph** - Workflow management
- **OpenAI** - GPT-4o-mini for LLM, text-embedding-3-small for embeddings
- **FAISS** - Vector database for semantic search
- **Tavily** - Real-time web search API
- **Langfuse** - Observability & prompt management
- **Streamlit** - Web interface
- **PyMuPDF** - Fast PDF parsing (with pdfplumber fallback)

## 📄 Supported File Formats

- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- Plain Text (`.txt`)

## 🤝 Collaboration Guide

### For Teammates

1. **Get the code**
```bash
git clone https://github.com/divyeshmutha12/AI-Interview-Agent.git
cd AI-Interview-Agent
```

2. **Create your own branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b your-name/feature-description
```

3. **Make changes and commit**
```bash
git add .
git commit -m "Description of changes"
```

4. **Push your branch**
```bash
git push origin feature/your-feature-name
```

5. **Create Pull Request on GitHub**
- Go to the repository on GitHub
- Click "Pull Requests" → "New Pull Request"
- Select your branch and create PR

### Workflow

1. Always pull latest changes before starting:
```bash
git checkout main
git pull origin main
```

2. Create a new branch for each feature:
```bash
git checkout -b feature/evaluation-improvements
```

3. Make small, focused commits:
```bash
git add specific_file.py
git commit -m "Add vector search integration"
```

4. Push and create PR for review:
```bash
git push origin feature/evaluation-improvements
```

## 🔐 Security Notes

- **Never commit `.env` file** - it contains sensitive API keys
- Use `.env.example` as a template for teammates
- Revoke and regenerate API keys if accidentally exposed
- GitHub has push protection - it will block pushes with secrets

## 📊 Monitoring

View traces and logs at: [Langfuse Dashboard](https://cloud.langfuse.com)

---

## 🐛 Troubleshooting

### Knowledge Base Issues
If KB Search returns no results:
1. **Check if documents are indexed:**
   - Go to "📚 Knowledge Base" tab in web UI
   - View statistics at top (should show document count > 0)
2. **Add documents:**
   - Upload documents via "📚 Knowledge Base" tab
   - Select appropriate domain
   - Verify successful indexing message
3. **Test search:**
   - Use search feature in "📚 Knowledge Base" tab
   - Try different queries or domains

### Tavily API Issues
If web search isn't working:
1. Check that `TAVILY_API_KEY` is set in `.env`
2. Verify API key is valid at [tavily.com](https://tavily.com)
3. Check API quota (1000 free searches/month)
4. System will automatically fall back to LLM simulation if Tavily unavailable

### Langfuse Connection Errors
If you see OpenTelemetry errors, these are non-blocking. The app will continue to work without telemetry.

### Import Errors
Make sure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Document Parsing Issues
Ensure you have the correct libraries:
```bash
pip install pymupdf pdfplumber python-docx
```
