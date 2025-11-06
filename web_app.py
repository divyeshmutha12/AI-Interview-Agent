"""
AI Interviewer System + Knowledge Base Builder (Merged Streamlit UI)
"""

import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- IMPORT SERVICES ----
from ai_interviewer.agents.agent_service import AgentService
from ai_interviewer.utils.document_parser import DocumentParser
from ai_interviewer.kb.kb_manager import create_kb, BASE_DIR

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit Page Config
st.set_page_config(page_title="AI Interviewer System", page_icon="🎯", layout="wide")

# -------- INITIALIZE SERVICES --------
@st.cache_resource
def initialize_services():
    try:
        agent_service = AgentService()
        doc_parser = DocumentParser(prefer_pdfplumber=True)
        return agent_service, doc_parser, None
    except Exception as e:
        return None, None, str(e)

def save_uploaded_file(uploaded_file) -> str:
    temp_dir = Path(tempfile.gettempdir()) / "ai_interviewer_uploads"
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

# -------------------------------------------
# MAIN APP
# -------------------------------------------
def main():

    st.markdown("<h1 style='text-align:center;'>🎯 AI Interviewer System</h1>", unsafe_allow_html=True)

    # Initialize Agents
    with st.spinner("Initializing AI Agents..."):
        agent_service, doc_parser, init_error = initialize_services()

    if init_error:
        st.error(f"Initialization Failed: {init_error}")
        st.stop()

    # Sidebar Config
    with st.sidebar:
        st.header("⚙️ Settings")
        session_id = st.text_input("Session ID (optional)")
        num_questions = st.slider("Number of Interview Questions", 1, 20, 5)
        st.info(f"Using Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Generate",
        "✅ Evaluate Answer",
        "ℹ️ About",
        "📚 Create KB"
    ])

    # ===== TAB 1: Upload & Generate =====
    with tab1:
        st.header("Upload Resume & Job Description")
        col1, col2 = st.columns(2)

        with col1:
            resume_file = st.file_uploader("Upload Resume", type=['pdf','docx','txt'])
        with col2:
            jd_file = st.file_uploader("Upload Job Description", type=['pdf','docx','txt'])

        if st.button("🚀 Generate Interview Questions", use_container_width=True):
            if not resume_file or not jd_file:
                st.error("Upload both Resume & Job Description.")
            else:
                resume_path = save_uploaded_file(resume_file)
                jd_path = save_uploaded_file(jd_file)

                docs = doc_parser.parse_resume_and_jd(resume_path, jd_path)
                resume_text = docs['resume']
                jd_text = docs['job_description']

                result = agent_service.generate_interview_questions(
                    candidate_resume=resume_text,
                    job_profile=jd_text,
                    num_questions=num_questions,
                    session_id=session_id or None
                )

                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success("✅ Questions Generated!")
                    st.write(result['result'])
                    st.download_button("📥 Download", result['result'], "interview_questions.txt")

    # ===== TAB 2: Evaluate Candidate Answer =====
    with tab2:
        st.header("Evaluate Candidate Answer")
        question_text = st.text_area("Interview Question")
        answer_text = st.text_area("Candidate Answer")

        if st.button("🔍 Evaluate"):
            if not question_text or not answer_text:
                st.warning("Enter both question and answer.")
            else:
                eval_req = f"Question: {question_text}\nAnswer: {answer_text}"
                result = agent_service.run(user_input=eval_req, session_id=session_id or None)
                st.write(result['result'])

    # ===== TAB 3: About =====
    with tab3:
        st.header("About System")
        st.write("AI-powered multi-agent interviewer and knowledge-based evaluation system.")

    # ===== TAB 4: Create KB =====
    with tab4:
        st.header("📚 Create Knowledge Base")
        kb_name = st.text_input("Knowledge Base Name")
        role_tag = st.text_input("Role Tag (e.g. java_backend, data_science)")
        uploaded_docs = st.file_uploader("Upload Training Documents", accept_multiple_files=True)

        if st.button("Create KB"):
            if not kb_name or not role_tag or not uploaded_docs:
                st.warning("Fill all fields.")
            else:
                create_kb(kb_name, role_tag, uploaded_docs)
                st.success(f"✅ KB '{kb_name}' created successfully!")

if __name__ == "__main__":
    main()
