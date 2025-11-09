"""
AI Interviewer Web Interface

Streamlit-based web application for uploading resumes and job descriptions,
generating interview questions, and evaluating candidate answers.
"""

import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_interviewer.agents.agent_service import AgentService
from ai_interviewer.utils.document_parser import DocumentParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Interviewer System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_services():
    """Initialize agent service and document parser (cached)"""
    try:
        agent_service = AgentService()
        doc_parser = DocumentParser()  # Uses PyMuPDF by default
        return agent_service, doc_parser, None
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return None, None, str(e)


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "ai_interviewer_uploads"
        temp_dir.mkdir(exist_ok=True)

        # Save file
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise


def main():
    """Main application"""

    # Header
    st.markdown('<h1 class="main-header">🎯 AI Interviewer System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload Resume & Job Description to Generate Interview Questions</p>',
        unsafe_allow_html=True
    )

    # Initialize services
    with st.spinner("Initializing AI agents..."):
        agent_service, doc_parser, init_error = initialize_services()

    if init_error:
        st.error(f"Failed to initialize services: {init_error}")
        st.info("Please check your .env file and ensure all API keys are set correctly.")
        st.stop()

    # Sidebar
    session_id = None  # Session tracking disabled

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Number of questions
        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=20,
            value=5,
            help="How many interview questions to generate"
        )

        # Model info
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        st.info(f"**Model:** {model}")

        # Supported formats
        st.subheader("📄 Supported Formats")
        formats = doc_parser.get_supported_formats()
        st.write(", ".join(formats))

        # Langfuse link
        st.subheader("📊 Monitoring")
        st.markdown("[View Traces in Langfuse](https://cloud.langfuse.com)")

    # Main tabs
    tab1, tab2 = st.tabs(["📤 Upload & Generate", "✅ Evaluate Answer"])

    # ===== TAB 1: Upload & Generate =====
    with tab1:
        st.header("Upload Resume & Job Description")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Resume")
            resume_file = st.file_uploader(
                "Upload Resume",
                type=['pdf', 'docx', 'txt'],
                help="Upload candidate's resume in PDF, DOCX, or TXT format",
                key="resume_upload"
            )

            if resume_file:
                st.success(f"✅ Uploaded: {resume_file.name}")
                st.caption(f"Size: {resume_file.size / 1024:.1f} KB")

        with col2:
            st.subheader("💼 Job Description")
            jd_file = st.file_uploader(
                "Upload Job Description",
                type=['pdf', 'docx', 'txt'],
                help="Upload job description in PDF, DOCX, or TXT format",
                key="jd_upload"
            )

            if jd_file:
                st.success(f"✅ Uploaded: {jd_file.name}")
                st.caption(f"Size: {jd_file.size / 1024:.1f} KB")

        st.markdown("---")

        # Generate button
        if st.button("🚀 Generate Interview Questions", type="primary", use_container_width=True):
            if not resume_file or not jd_file:
                st.error("⚠️ Please upload both Resume and Job Description")
            else:
                try:
                    with st.spinner("Processing documents..."):
                        # Save uploaded files
                        resume_path = save_uploaded_file(resume_file)
                        jd_path = save_uploaded_file(jd_file)

                        # Parse documents
                        st.info("📖 Extracting text from documents...")
                        docs = doc_parser.parse_resume_and_jd(resume_path, jd_path)

                        resume_text = docs['resume']
                        jd_text = docs['job_description']

                        # Show preview
                        with st.expander("📝 Preview Extracted Text"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.text_area(
                                    "Resume Preview",
                                    resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
                                    height=200
                                )
                            with col2:
                                st.text_area(
                                    "JD Preview",
                                    jd_text[:500] + "..." if len(jd_text) > 500 else jd_text,
                                    height=200
                                )

                    # Generate questions
                    with st.spinner("🤖 AI agents are generating interview questions..."):
                        result = agent_service.generate_interview_questions(
                            candidate_resume=resume_text,
                            job_profile=jd_text,
                            num_questions=num_questions,
                            session_id=None
                        )

                    # Display results
                    if result.get('error'):
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.success("✅ Interview questions generated successfully!")

                        # Metadata
                        with st.expander("ℹ️ Generation Details"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**Pipeline:**")
                                st.text(result.get('pipeline', 'N/A'))
                            with col2:
                                st.markdown("**Tool Used:**")
                                st.text(result.get('tool_used', 'N/A'))
                            with col3:
                                st.markdown("**Model:**")
                                st.text(result['metadata']['model'])

                        # Candidate Analysis (Explainable AI)
                        if result.get('candidate_analysis'):
                            with st.expander("🔍 Candidate Analysis (How Agent Analyzed the Resume against JD)", expanded=False):
                                st.markdown("""
                                **This shows how the AI analyzed the candidate's resume to generate personalized questions.**
                                """)
                                st.code(result['candidate_analysis'], language='text')
                                st.caption("💡 This analysis helps ensure questions are tailored to the candidate's actual experience")

                        # Questions
                        st.markdown("---")
                        st.subheader("📋 Generated Interview Questions")
                        st.markdown(result['result'])

                        # Download option
                        st.download_button(
                            label="📥 Download Questions",
                            data=result['result'],
                            file_name="interview_questions.txt",
                            mime="text/plain"
                        )

                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    logger.error(f"Processing error: {e}", exc_info=True)

    # ===== TAB 2: Evaluate Answer =====
    with tab2:
        st.header("Evaluate Candidate Answer")

        st.markdown("""
        Enter a question and the candidate's answer below to get an AI-powered evaluation
        with scoring and feedback.
        """)

        question_text = st.text_area(
            "Interview Question",
            height=100,
            placeholder="e.g., What is a closure in Python?",
            help="Enter the interview question that was asked"
        )

        answer_text = st.text_area(
            "Candidate's Answer",
            height=200,
            placeholder="Enter the candidate's response here...",
            help="Paste or type the candidate's answer"
        )

        if st.button("🔍 Evaluate Answer", type="primary", use_container_width=True):
            if not question_text or not answer_text:
                st.error("⚠️ Please provide both question and answer")
            else:
                try:
                    # Format evaluation request
                    eval_request = f"""
                    Question: {question_text}

                    Answer: {answer_text}
                    """

                    with st.spinner("🤖 AI agents are evaluating the answer..."):
                        result = agent_service.run(
                            user_input=eval_request,
                            session_id=None
                        )

                    # Display results
                    if result.get('error'):
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.success("✅ Evaluation complete!")

                        # Metadata
                        with st.expander("ℹ️ Evaluation Details"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**Pipeline:**")
                                st.text(result.get('pipeline', 'N/A'))
                            with col2:
                                st.markdown("**Tool Used:**")
                                st.text(result.get('tool_used', 'N/A'))
                            with col3:
                                st.markdown("**Model:**")
                                st.text(result['metadata']['model'])

                        # Evaluation results
                        st.markdown("---")
                        st.subheader("📊 Evaluation Results")
                        st.markdown(result['result'])

                except Exception as e:
                    st.error(f"❌ Error evaluating answer: {str(e)}")
                    logger.error(f"Evaluation error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
