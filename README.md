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
OPENAI_API_KEY=your_actual_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_PROMPT_LABEL=your_name
```

**⚠️ NEVER commit `.env` file!**

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
│   │   └── document_parser.py       # PDF/DOCX/TXT parsing
│   └── run_interview.py             # CLI interface
├── web_app.py                       # Streamlit web interface
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
└── README.md                        # This file
```

## 🔧 Tech Stack

- **LangChain** - Agent orchestration
- **LangGraph** - Workflow management
- **OpenAI** - GPT-4o-mini for LLM
- **Langfuse** - Observability & prompt management
- **Streamlit** - Web interface
- **PyPDF2/pdfplumber** - Document parsing

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

## 🐛 Troubleshooting

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
pip install PyPDF2 pdfplumber python-docx
```

## 👥 Team

- **Divyesh** - Lead Developer

## 📝 License

[Add your license here]

---

Made with ❤️ using Claude Code
