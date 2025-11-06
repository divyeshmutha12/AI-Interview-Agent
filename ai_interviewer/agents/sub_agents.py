"""
Sub-Agents for AI Interviewer System (Proper Multi-Agent Architecture)

This module implements a true multi-agent hierarchy:
1. Supervisor Agent
   └── Question & Answer Generator Agent (LangChain Agent)
       ├── Candidate Analysis Agent (LangChain Agent with tools)
       ├── KB Search Agent (Deep Agent with tools)
       └── Web Search Agent (Deep Agent with tools)
   └── Evaluator Pipeline

All agents use LangChain's create_agent() pattern.
"""
import os
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, Any, Optional, List, Callable
import logging
from docx import Document
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
import json

logger = logging.getLogger(__name__)


# ==========================================
# FALLBACK PROMPTS
# ==========================================

FALLBACK_PROMPTS = {
    # Candidate Analysis Agent and its tools
    "candidate_analysis_agent": """
You are a Candidate Analysis Agent for an AI Interviewer system.

You have access to the following tools:
1. parse_resume - Extract skills, experience, and education from resume
2. parse_job_description - Extract requirements and responsibilities from job profile
3. compare_profiles - Match candidate qualifications to job requirements

Your job:
1. FIRST call parse_resume with the candidate's resume
2. THEN call parse_job_description with the job profile
3. FINALLY call compare_profiles to analyze fit

Output a structured analysis with:
- Experience level (junior/mid/senior/lead)
- Skill match percentage
- Recommended question difficulty
- Focus areas for interview
""",
    "parse_resume": """Extract structured information from candidate resume.
Return JSON with: skills, experience_years, education, domain, certifications.""",

    "parse_job_description": """Extract structured information from job description.
Return JSON with: required_skills, preferred_skills, experience_required, responsibilities, technologies.""",

    "compare_profiles": """Compare candidate resume data with job requirements.
Analyze skill gaps, experience match, and provide interview focus areas.""",

    # KB Search Agent and its tools
     "kb_search_agent": """
You are a highly intelligent and specialized Knowledge Base Search Agent for an AI Interviewer system. Your mission is to retrieve the most relevant, diverse, and comprehensive information from available knowledge bases to assist in generating precise interview questions for *any* job role.

You have access to the following powerful tools, each designed for a specific type of information retrieval:

1.  **list_available_knowledge_bases()**:
    *   **Purpose**: Discover which knowledge bases are currently available. You **MUST** call this tool first to understand your searchable domains.
    *   **Returns**: A JSON string listing available KBs and their general roles (e.g., `[{"name": "Openshift ", "role": "Cloud Platform"}, {"name": "AI&ML", "role": "Artificial Intelligence & Machine Learning"}]`).
    *   **How to Use**: `list_available_knowledge_bases()`

2.  **vector_search(input_json: str)**:
    *   **Purpose**: Ideal for **semantic search** and exploring broad concepts, definitions, high-level principles, or when the exact phrasing isn't known. Use this when you need content that is *conceptually similar* to your query, even if the keywords don't precisely match. Excellent for understanding "what is X" or "how does Y work" at a deeper level.
    *   **Input**: A JSON string with two fields:
        *   `kb_name` (string): The name of the knowledge base to search (e.g., "Openshift ", "AI&ML").
        *   `query_text` (string): The conceptual question or topic you want to find semantically similar information for.
    *   **How to Use**: `vector_search('{"kb_name": "Openshift ", "query_text": "Explain Kubernetes pod scheduling."}')`

3.  **keyword_search(input_json: str)**:
    *   **Purpose**: Best for finding content related to **specific named entities, technologies, versions, frameworks, commands, or exact terms**. This tool is optimized to surface documents containing or closely related to the *precise keywords* you provide. Use this when you have very specific terms you want to match, like "OpenShift CLI commands", "Kubernetes networking policies", or "Python troubleshooting".
    *   **Input**: A JSON string with two fields:
        *   `kb_name` (string): The name of the knowledge base to search.
        *   `keywords_list` (string): A comma-separated string of exact keywords or technical terms you are looking for.
    *   **How to Use**: `keyword_search('{"kb_name": "Openshift ", "keywords_list": "oc debug, StatefulSets, Ingress Controller"}' )`

4.  **topic_lookup(input_json: str)**:
    *   **Purpose**: Designed to retrieve **structured question templates, common interview topics, or pre-curated content blocks** associated with recognized subject areas. Use this when you can clearly identify a well-defined subject category (e.g., "Containerization", "Networking fundamentals", "Troubleshooting scenarios", "System Design patterns").
    *   **Input**: A JSON string with two fields:
        *   `kb_name` (string): The name of the knowledge base to search.
        *   `topic_name` (string): The name of the topic category you are interested in.
    *   **How to Use**: `topic_lookup('{"kb_name": "Openshift ", "topic_name": "OpenShift Installation & Configuration"}' )`

Your Job Workflow for any search request from the Question & Answer Generator Agent:

1.  **Understand the Request**: Carefully analyze the incoming request (e.g., candidate's skills, job requirements, desired focus areas).
2.  **Identify Relevant Knowledge Base(s)**: First, call `list_available_knowledge_bases()` to see your options. Based on the job role and candidate profile, select the most appropriate KB (e.g., "Openshift " for an OpenShift role).
3.  **Strategic Multi-Tool Application**: Plan your search by considering what type of information each tool is best suited for. You **MUST strive to use a combination of these tools** to get a comprehensive understanding:
    *   **Start with `topic_lookup`**: If the request clearly points to specific subject areas (e.g., "networking", "security", "troubleshooting"), use `topic_lookup` to get foundational questions or templates.
    *   **Then use `keyword_search`**: For explicit technologies, commands, or specific features mentioned in the resume or job description (e.g., "Prometheus", "Helm charts", "oc rsh", "Python", "GPT-3"), use `keyword_search`.
    *   **Complement with `vector_search`**: For broader conceptual understanding, definitions, architectural principles, or related ideas that might not be exact keywords or topics (e.g., "scalability patterns in OpenShift", "ethical considerations in AI", "best practices for cloud migration"), use `vector_search`.
    *   **Iterate**: If one tool doesn't yield sufficient results, try rephrasing your query for another tool, or use a different tool altogether for the same concept.
4.  **Synthesize and Structure**: Combine the diverse results from all successful tool calls into a unified, coherent, and highly informative output. Do not just return raw tool outputs. Ensure the information is well-organized and directly supports the generation of varied interview questions (technical, conceptual, scenario-based).

Your final output should be a rich compilation of relevant questions, topics, concepts, and technical details to empower the Question & Answer Generator.
""",

    # Web Search Agent and its tools
    "web_search_agent": """
You are a Web Search Agent (Deep Agent) for researching current industry trends.

You have access to the following tools:
1. web_search - Search the web for current information
2. extract_trends - Extract technology trends from search results
3. get_latest_info - Get latest information about specific technologies

Your job:
1. Research current trends in the candidate's domain
2. Find latest technologies and best practices
3. Identify current industry requirements

Return up-to-date information to inform modern interview questions.
""",
    "web_search": """Search the web for current information.
Input: search query. Returns: relevant web content and links.""",

    "extract_trends": """Analyze content to extract technology trends.
Input: content text. Returns: trending topics and technologies.""",

    "get_latest_info": """Get latest information about specific technology.
Input: technology name. Returns: current version, best practices, usage.""",

    # Question & Answer Generator Agent
    "question_answer_generator": """
You are the Question & Answer Generator Agent for an AI Interviewer System.

You have access to 3 intelligent sub-agents:
1. candidate_analysis - Analyzes candidate resume vs job requirements (AGENT with 3 tools)
2. kb_search - Searches knowledge base for relevant content (DEEP AGENT with 3 tools)
3. web_search - Researches current trends and technologies (DEEP AGENT with 3 tools)

IMPORTANT: These are not simple tools - they are intelligent agents that can use their own tools autonomously!

## Input Format:

You will receive a request with:
- **Resume**: Full candidate resume/CV text
- **Job Profile**: Full job description/requirements
- **Number of Questions**: How many questions to generate

## Workflow:

### Step 1: Candidate Analysis
CALL the candidate_analysis agent with BOTH the resume and job profile:
```
Analyze this candidate for the job.

Resume:
[full resume text]

Job Profile:
[full job description]
```

The candidate_analysis agent will:
- Parse the resume (using parse_resume tool)
- Parse the job description (using parse_job_description tool)
- Compare and analyze fit (using compare_profiles tool)
- Return analysis with experience level, skills match, recommended difficulty

### Step 2: Knowledge Base Search
Based on the candidate analysis, CALL the kb_search agent:
```
Find relevant interview questions for:
- Experience level: [from analysis]
- Skills: [from analysis]
- Domain: [from analysis]
- Focus areas: [from analysis]
```

The kb_search agent will:
- Use vector search for semantic similarity
- Use keyword search for specific technologies
- Use topic lookup for question templates

### Step 3: Web Research (Optional)
Optionally CALL the web_search agent for current trends:
```
Research current trends for:
- Technologies: [from resume/job]
- Industry: [from job profile]
```

### Step 4: Generate Questions with Ideal Answers
Based on all gathered information, generate exactly N questions (where N is from the request).

## Output Format:

Provide a structured response with:

1. **Candidate Analysis Summary**: Brief summary of candidate-job fit
2. **Questions**: Array of exactly N questions
3. **Metadata**: Information about generation

Example Output:
```
## Candidate Analysis
[Summary of candidate's experience level, skills match, and focus areas]

## Interview Questions

### Question 1: [Topic Area]
**Type**: Technical/Behavioral/Scenario
**Difficulty**: Easy/Medium/Hard

**Question:**
[Question text]

**Ideal Answer:**
[Comprehensive ideal answer with key points that should be covered]

**Key Concepts**: concept1, concept2, concept3

---

### Question 2: [Topic Area]
...

## Metadata
- Candidate Level: junior/mid/senior
- Total Questions Generated: N
- Focus Areas: area1, area2, area3
- Sources Used: candidate_analysis, kb_search, web_search
```

## Important Rules:

1. **ALWAYS call candidate_analysis FIRST** with both resume and job profile
2. **ALWAYS call kb_search** based on the analysis
3. **Generate EXACTLY the number of questions requested**
4. **Include ideal answers for EVERY question** (this is critical for evaluation)
5. **Match difficulty to candidate's level** (from analysis)
6. **Focus on areas identified in the analysis**

REMEMBER: Use your intelligent sub-agents - they will handle the complex analysis for you!
""",

    # Evaluation Pipeline (unchanged)
    "topic_extraction": """Extract main topics, technical concepts, and key points from candidate answers.""",
    "ideal_generation": """Generate reference answers with key points, technical requirements, and critical concepts.""",
    "topic_evaluator": """Compare candidate answers against ideal answers and provide scores (0-10) per topic.""",
    "cross_validation": """Review evaluations, check consistency, and provide final aggregate scores and feedback."""
}


# ==========================================
# CANDIDATE ANALYSIS AGENT & TOOLS
# ==========================================

def create_candidate_analysis_agent(
    llm: ChatOpenAI,
    langfuse_client: Optional[Langfuse] = None,
    langfuse_handler: Optional[CallbackHandler] = None,
    prompt_label: str = "production"
):
    """
    Create Candidate Analysis Agent (LangChain Agent).

    This agent has 3 tools to analyze candidates:
    - parse_resume
    - parse_job_description
    - compare_profiles
    """
    logger.info("Creating Candidate Analysis Agent...")

    # Create tools for this agent
    @tool
    def parse_resume(resume_text: str) -> str:
        """
        Extract structured information from candidate resume.

        Extracts:
        - Technical skills and proficiency levels
        - Years of experience
        - Education and certifications
        - Domain expertise
        - Previous roles and responsibilities

        Args:
            resume_text: Full text of candidate's resume

        Returns:
            JSON string with structured resume data
        """
        try:
            prompt = FALLBACK_PROMPTS.get("parse_resume", "")
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Resume:\n{resume_text}\n\nExtract structured information."}
            ]

            if langfuse_handler:
                config = RunnableConfig(
                    callbacks=[langfuse_handler],
                    tags=["parse_resume", "candidate_analysis_tool"],
                    metadata={"tool_type": "parse_resume"}
                )
                response = llm.invoke(messages, config=config)
            else:
                response = llm.invoke(messages)

            logger.info("Resume parsed successfully")
            return response.content
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            return f"Error: {str(e)}"

    @tool
    def parse_job_description(job_profile: str) -> str:
        """
        Extract structured information from job description.

        Extracts:
        - Required skills and qualifications
        - Preferred skills
        - Years of experience required
        - Responsibilities and duties
        - Technologies and tools needed

        Args:
            job_profile: Full text of job description

        Returns:
            JSON string with structured job requirements
        """
        try:
            prompt = FALLBACK_PROMPTS.get("parse_job_description", "")
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Job Profile:\n{job_profile}\n\nExtract requirements."}
            ]

            if langfuse_handler:
                config = RunnableConfig(
                    callbacks=[langfuse_handler],
                    tags=["parse_job_description", "candidate_analysis_tool"],
                    metadata={"tool_type": "parse_job_description"}
                )
                response = llm.invoke(messages, config=config)
            else:
                response = llm.invoke(messages)

            logger.info("Job description parsed successfully")
            return response.content
        except Exception as e:
            logger.error(f"Error parsing job description: {e}")
            return f"Error: {str(e)}"

    @tool
    def compare_profiles(resume_data: str, job_data: str) -> str:
        """
        Compare candidate qualifications with job requirements.

        Analyzes:
        - Skill match percentage
        - Experience level fit
        - Gap analysis
        - Recommended interview focus areas
        - Question difficulty level

        Args:
            resume_data: Structured resume data from parse_resume
            job_data: Structured job requirements from parse_job_description

        Returns:
            Analysis report with match score and recommendations
        """
        try:
            prompt = FALLBACK_PROMPTS.get("compare_profiles", "")
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Resume Data:\n{resume_data}\n\nJob Requirements:\n{job_data}\n\nAnalyze fit."}
            ]

            if langfuse_handler:
                config = RunnableConfig(
                    callbacks=[langfuse_handler],
                    tags=["compare_profiles", "candidate_analysis_tool"],
                    metadata={"tool_type": "compare_profiles"}
                )
                response = llm.invoke(messages, config=config)
            else:
                response = llm.invoke(messages)

            logger.info("Profile comparison completed")
            return response.content
        except Exception as e:
            logger.error(f"Error comparing profiles: {e}")
            return f"Error: {str(e)}"

    # Create the tools list
    tools = [parse_resume, parse_job_description, compare_profiles]

    # Load system prompt from Langfuse or use fallback
    system_prompt = FALLBACK_PROMPTS["candidate_analysis_agent"]
    if langfuse_client:
        try:
            prompt_obj = langfuse_client.get_prompt(
                "AIInterviewer/qg/candidate_analysis_agent",
                label=prompt_label
            )
            system_prompt = prompt_obj.compile()
            logger.info("Loaded candidate_analysis_agent prompt from Langfuse")
        except Exception as e:
            logger.warning(f"Failed to load Langfuse prompt: {e}")

    # Create agent using LangChain's create_agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    logger.info(f"Candidate Analysis Agent created with {len(tools)} tools")
    return agent


# ==========================================
# KB SEARCH AGENT & TOOLS (Deep Agent)
# ==========================================
from functools import lru_cache
# Base directory for all knowledge bases
BASE_KNOWLEDGE_BASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'knowledge_bases') # Ensure this matches your project structure

def get_faiss_index_path(kb_name: str) -> str:
    """Constructs the path to the FAISS index for a given knowledge base name."""
    sanitized_kb_name = kb_name.replace(" ", "_").lower()
    return os.path.join(BASE_KNOWLEDGE_BASES_DIR, sanitized_kb_name, "faiss_index")

def create_kb_search_agent(
    llm: ChatOpenAI,
    langfuse_client: Optional[Langfuse] = None,
    langfuse_handler: Optional[CallbackHandler] = None,
    prompt_label: str = "production"
):
    """
    Create KB Search Agent (Deep Agent with multiple search tools).

    This agent has 3 tools for knowledge base search:
    1. list_available_knowledge_bases (to discover KBs)
    2. vector_search (semantic similarity on a chosen KB)
    3. keyword_search (exact match on a chosen KB)
    4. topic_lookup (category-based on a chosen KB)
    """
    logger.info("Creating KB Search Agent (Deep Agent) with dynamic KB selection...")

    # Initialize Embeddings once for use by search tools
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    @lru_cache(maxsize=10)
    # Helper to load a specific vector store when requested by a tool
    def _load_vector_store(kb_name: str):
        faiss_index_path = get_faiss_index_path(kb_name)
        if os.path.exists(faiss_index_path):
            try:
                vector_store = FAISS.load_local(
                    faiss_index_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS index loaded successfully from: {faiss_index_path} for KB '{kb_name}'")
                return vector_store
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {faiss_index_path} for KB '{kb_name}': {e}")
                return None
        else:
            logger.warning(f"FAISS index not found at {faiss_index_path} for KB '{kb_name}'.")
            return None

    @tool
    def list_available_knowledge_bases() -> str:
        """
        Lists all available knowledge bases by inspecting the `knowledge_bases` directory.
        Returns a JSON string with a list of {'name': 'KB Name', 'role': 'KB Role'} for each KB.
        """
        available_kbs = []
        kb_dirs = [d for d in os.listdir(BASE_KNOWLEDGE_BASES_DIR) if os.path.isdir(os.path.join(BASE_KNOWLEDGE_BASES_DIR, d))]

        for kb_dir_name in kb_dirs:
            kb_full_path = os.path.join(BASE_KNOWLEDGE_BASES_DIR, kb_dir_name)
            metadata_path = os.path.join(kb_full_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        available_kbs.append({"name": metadata.get("name", kb_dir_name), "role": metadata.get("role", "N/A")})
                except Exception as e:
                    logger.warning(f"Could not read metadata for KB '{kb_dir_name}': {e}")
            else:
                available_kbs.append({"name": kb_dir_name, "role": "No metadata"})

        logger.info(f"Listed {len(available_kbs)} knowledge bases.")
        return json.dumps(available_kbs)


    # Create tools for this agent
    @tool
    def vector_search(input_json: str, k: int = 5) -> str:
        """
        Search a specific vector database using semantic similarity.
        The input must be a JSON string with 'kb_name' and 'query_text' fields.

        Finds questions and topics semantically similar to the query.
        Uses embeddings to find related content even with different wording.

        Args:
            input_json: JSON string with 'kb_name' (e.g., "ai_&_ml") and 'query_text'
            k: Number of relevant chunks to retrieve (default: 5)

        Returns:
            Similar questions and topics from knowledge base,
            or an error message if the knowledge base is not available.
        """
        try:
            input_data = json.loads(input_json)
            kb_name = input_data.get('kb_name')
            query = input_data.get('query_text')

            if not kb_name or not query:
                return "Error: Both 'kb_name' and 'query_text' must be provided in the input JSON."

            vector_store = _load_vector_store(kb_name)
            if vector_store is None:
                return f"Error: Knowledge base '{kb_name}' not found or could not be loaded."

            # Perform actual vector search
            docs = vector_store.similarity_search(query, k=k)
            results = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Vector search completed for query: '{query}' in KB: '{kb_name}'")
            return results
        except json.JSONDecodeError:
            return "Error: Invalid JSON input for vector_search."
        except Exception as e:
            logger.error(f"Error in vector search with FAISS for KB '{kb_name}': {e}")
            return f"Error performing vector search in KB '{kb_name}': {str(e)}"

    @tool
    def keyword_search(input_json: str, k: int = 5) -> str:
        """
        Search a specific knowledge base by exact keywords or semantic similarity emphasizing keywords.
        The input must be a JSON string with 'kb_name' and 'keywords_list' fields.

        Finds content matching specific keywords or technologies.
        Use for precise technology names, frameworks, or concepts.
        Currently leverages vector search for a more robust keyword match.

        Args:
            input_json: JSON string with 'kb_name' (e.g., "ai_&_ml") and 'keywords_list' (comma-separated string)
            k: Number of relevant chunks to retrieve (default: 5)

        Returns:
            Matching content from knowledge base.
        """
        try:
            input_data = json.loads(input_json)
            kb_name = input_data.get('kb_name')
            keywords = input_data.get('keywords_list')

            if not kb_name or not keywords:
                return "Error: Both 'kb_name' and 'keywords_list' must be provided in the input JSON."

            vector_store = _load_vector_store(kb_name)
            if vector_store is None:
                return f"Error: Knowledge base '{kb_name}' not found or could not be loaded."

            docs = vector_store.similarity_search(keywords, k=k) # Using vector search for keyword matching
            results = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Keyword search completed for keywords: '{keywords}' in KB: '{kb_name}'")
            return results
        except json.JSONDecodeError:
            return "Error: Invalid JSON input for keyword_search."
        except Exception as e:
            logger.error(f"Error in keyword search with FAISS for KB '{kb_name}': {e}")
            return f"Error performing keyword search in KB '{kb_name}': {str(e)}"

    @tool
    def topic_lookup(input_json: str, k: int = 5) -> str:
        """
        Find question templates or content by topic category using semantic search within a specific knowledge base.
        The input must be a JSON string with 'kb_name' and 'topic_name' fields.

        Retrieves pre-defined question templates or content for specific topics.
        Topics include: algorithms, databases, system design, OOP, etc.

        Args:
            input_json: JSON string with 'kb_name' (e.g., "ai_&_ml") and 'topic_name'
            k: Number of relevant chunks to retrieve (default: 5)

        Returns:
            Question templates and examples for the topic from knowledge base.
        """
        try:
            input_data = json.loads(input_json)
            kb_name = input_data.get('kb_name')
            topic = input_data.get('topic_name')

            if not kb_name or not topic:
                return "Error: Both 'kb_name' and 'topic_name' must be provided in the input JSON."

            vector_store = _load_vector_store(kb_name)
            if vector_store is None:
                return f"Error: Knowledge base '{kb_name}' not found or could not be loaded."

            docs = vector_store.similarity_search(f"Interview questions on {topic}", k=k)
            results = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Topic lookup completed for topic: '{topic}' in KB: '{kb_name}'")
            return results
        except json.JSONDecodeError:
            return "Error: Invalid JSON input for topic_lookup."
        except Exception as e:
            logger.error(f"Error in topic lookup with FAISS for KB '{kb_name}': {e}")
            return f"Error performing topic lookup in KB '{kb_name}': {str(e)}"

    # Create the tools list - now includes list_available_knowledge_bases
    tools = [list_available_knowledge_bases, vector_search, keyword_search, topic_lookup]

    # Load system prompt from Langfuse or use fallback
    system_prompt = FALLBACK_PROMPTS["kb_search_agent"]
    if langfuse_client:
        try:
            prompt_obj = langfuse_client.get_prompt(
                "AIInterviewer/qg/kb_search_agent",
                label=prompt_label
            )
            system_prompt = prompt_obj.compile()
            logger.info("Loaded kb_search_agent prompt from Langfuse")
        except Exception as e:
            logger.warning(f"Failed to load Langfuse prompt: {e}")

    # Create agent using LangChain's create_agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    logger.info(f"KB Search Agent created with {len(tools)} tools, with dynamic KB selection capability.")
    return agent


# ==========================================
# WEB SEARCH AGENT & TOOLS (Deep Agent)
# ==========================================

def create_web_search_agent(
    llm: ChatOpenAI,
    langfuse_client: Optional[Langfuse] = None,
    langfuse_handler: Optional[CallbackHandler] = None,
    prompt_label: str = "production"
):
    """
    Create Web Search Agent (Deep Agent with web research tools).

    This agent has 3 tools for web research:
    - web_search
    - extract_trends
    - get_latest_info
    """
    logger.info("Creating Web Search Agent (Deep Agent)...")

    # Create tools for this agent
    @tool
    def web_search(query: str) -> str:
        """
        Search the web for current information.

        Searches for latest information about technologies, trends, and best practices.
        Returns current web content relevant to the query.

        Args:
            query: Search query

        Returns:
            Relevant web content and links
        """
        try:
            # TODO: Implement actual web search with Tavily/DuckDuckGo
            # For now, simulate with LLM
            prompt = FALLBACK_PROMPTS.get("web_search", "")
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Search the web for: {query}"}
            ]

            if langfuse_handler:
                config = RunnableConfig(
                    callbacks=[langfuse_handler],
                    tags=["web_search", "web_search_tool", "deep_agent"],
                    metadata={"tool_type": "web_search"}
                )
                response = llm.invoke(messages, config=config)
            else:
                response = llm.invoke(messages)

            logger.info("Web search completed")
            return response.content
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return f"Error: {str(e)}"

    @tool
    def extract_trends(content: str) -> str:
        """
        Extract technology trends from content.

        Analyzes content to identify trending technologies, frameworks, and practices.
        Returns structured information about current industry trends.

        Args:
            content: Text content to analyze

        Returns:
            Trending topics and technologies
        """
        try:
            prompt = FALLBACK_PROMPTS.get("extract_trends", "")
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Extract trends from: {content}"}
            ]

            if langfuse_handler:
                config = RunnableConfig(
                    callbacks=[langfuse_handler],
                    tags=["extract_trends", "web_search_tool", "deep_agent"],
                    metadata={"tool_type": "extract_trends"}
                )
                response = llm.invoke(messages, config=config)
            else:
                response = llm.invoke(messages)

            logger.info("Trend extraction completed")
            return response.content
        except Exception as e:
            logger.error(f"Error extracting trends: {e}")
            return f"Error: {str(e)}"

    @tool
    def get_latest_info(technology: str) -> str:
        """
        Get latest information about a specific technology.

        Retrieves current version, best practices, and usage patterns.
        Includes latest updates and industry adoption.

        Args:
            technology: Technology name (e.g., "React", "Kubernetes")

        Returns:
            Latest information about the technology
        """
        try:
            prompt = FALLBACK_PROMPTS.get("get_latest_info", "")
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Get latest info for: {technology}"}
            ]

            if langfuse_handler:
                config = RunnableConfig(
                    callbacks=[langfuse_handler],
                    tags=["get_latest_info", "web_search_tool", "deep_agent"],
                    metadata={"tool_type": "get_latest_info"}
                )
                response = llm.invoke(messages, config=config)
            else:
                response = llm.invoke(messages)

            logger.info("Latest info retrieval completed")
            return response.content
        except Exception as e:
            logger.error(f"Error getting latest info: {e}")
            return f"Error: {str(e)}"

    # Create the tools list
    tools = [web_search, extract_trends, get_latest_info]

    # Load system prompt from Langfuse or use fallback
    system_prompt = FALLBACK_PROMPTS["web_search_agent"]
    if langfuse_client:
        try:
            prompt_obj = langfuse_client.get_prompt(
                "AIInterviewer/qg/web_search_agent",
                label=prompt_label
            )
            system_prompt = prompt_obj.compile()
            logger.info("Loaded web_search_agent prompt from Langfuse")
        except Exception as e:
            logger.warning(f"Failed to load Langfuse prompt: {e}")

    # Create agent using LangChain's create_agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    logger.info(f"Web Search Agent created with {len(tools)} tools")
    return agent


# ==========================================
# QUESTION & ANSWER GENERATOR AGENT
# ==========================================

def create_question_answer_generator_agent(
    llm: ChatOpenAI,
    langfuse_client: Optional[Langfuse] = None,
    langfuse_handler: Optional[CallbackHandler] = None,
    prompt_label: str = "production"
):
    """
    Create the Question & Answer Generator Agent.

    This agent has 3 SUB-AGENTS as tools:
    - Candidate Analysis Agent
    - KB Search Agent (Deep Agent)
    - Web Search Agent (Deep Agent)

    These sub-agents are intelligent and can use their own tools!
    """
    logger.info("Creating Question & Answer Generator Agent...")

    # Create the 3 sub-agents
    candidate_analysis_agent = create_candidate_analysis_agent(
        llm, langfuse_client, langfuse_handler, prompt_label
    )
    kb_search_agent = create_kb_search_agent(
        llm, langfuse_client, langfuse_handler, prompt_label
    )
    web_search_agent = create_web_search_agent(
        llm, langfuse_client, langfuse_handler, prompt_label
    )

    # Wrap each sub-agent as a tool
    @tool
    def candidate_analysis(input_data: str) -> str:
        """
        Analyze candidate resume against job requirements.

        This is an INTELLIGENT AGENT that:
        - Parses candidate resume
        - Parses job description
        - Compares and analyzes fit
        - Recommends interview focus areas

        Args:
            input_data: JSON with "resume" and "job_profile" fields

        Returns:
            Structured analysis of candidate-job fit
        """
        try:
            logger.info("Invoking Candidate Analysis Agent...")
            result = candidate_analysis_agent.invoke({
                "messages": [{"role": "user", "content": input_data}]
            })
            final_message = result["messages"][-1] if result.get("messages") else None
            output = final_message.content if final_message else "No analysis"
            logger.info("Candidate Analysis Agent completed")
            return output
        except Exception as e:
            logger.error(f"Error in candidate analysis agent: {e}")
            return f"Error: {str(e)}"

    @tool
    def kb_search(query: str) -> str:
        """
        Search knowledge base for relevant interview content.

        This is an INTELLIGENT DEEP AGENT that:
        - Uses vector search for semantic similarity
        - Uses keyword search for exact matches
        - Looks up question templates by topic

        Args:
            query: Search query based on candidate profile and requirements

        Returns:
            Relevant questions, topics, and concepts from knowledge base
        """
        try:
            logger.info("Invoking KB Search Agent...")
            result = kb_search_agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            final_message = result["messages"][-1] if result.get("messages") else None
            output = final_message.content if final_message else "No results"
            logger.info("KB Search Agent completed")
            return output
        except Exception as e:
            logger.error(f"Error in kb search agent: {e}")
            return f"Error: {str(e)}"

    @tool
    def web_search(query: str) -> str:
        """
        Research current industry trends and technologies.

        This is an INTELLIGENT DEEP AGENT that:
        - Searches the web for current information
        - Extracts technology trends
        - Gets latest information about specific technologies

        Args:
            query: Research query about trends or technologies

        Returns:
            Current trends and latest information
        """
        try:
            logger.info("Invoking Web Search Agent...")
            result = web_search_agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            final_message = result["messages"][-1] if result.get("messages") else None
            output = final_message.content if final_message else "No results"
            logger.info("Web Search Agent completed")
            return output
        except Exception as e:
            logger.error(f"Error in web search agent: {e}")
            return f"Error: {str(e)}"

    # Create tools list with the 3 wrapped agents
    tools = [candidate_analysis, kb_search, web_search]

    # Load system prompt from Langfuse or use fallback
    system_prompt = FALLBACK_PROMPTS["question_answer_generator"]
    if langfuse_client:
        try:
            prompt_obj = langfuse_client.get_prompt(
                "AIInterviewer/qg/question_answer_generator",
                label=prompt_label
            )
            system_prompt = prompt_obj.compile()
            logger.info("Loaded question_answer_generator prompt from Langfuse")
        except Exception as e:
            logger.warning(f"Failed to load Langfuse prompt: {e}")
            logger.info("Using fallback prompt")

    # Create agent using LangChain's create_agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    logger.info(f"Question & Answer Generator Agent created with {len(tools)} sub-agents as tools")
    return agent


# ==========================================
# EVALUATION PIPELINE (Unchanged)
# ==========================================

class SubAgent:
    """Base class for evaluation sub-agents"""

    def __init__(
        self,
        agent_name: str,
        prompt_id: str,
        llm: ChatOpenAI,
        langfuse_client: Optional[Langfuse] = None,
        langfuse_handler: Optional[CallbackHandler] = None,
        prompt_label: str = "production"
    ):
        self.agent_name = agent_name
        self.prompt_id = prompt_id
        self.llm = llm
        self.langfuse_client = langfuse_client
        self.langfuse_handler = langfuse_handler
        self.prompt_label = prompt_label
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load prompt from Langfuse or use fallback"""
        if self.langfuse_client:
            try:
                prompt_obj = self.langfuse_client.get_prompt(
                    self.prompt_id,
                    label=self.prompt_label
                )
                compiled_prompt = prompt_obj.compile()
                logger.info(f"Loaded prompt from Langfuse for {self.agent_name}")
                return compiled_prompt
            except Exception as e:
                logger.warning(f"Failed to load Langfuse prompt for {self.agent_name}: {e}")
                logger.info(f"Using fallback prompt for {self.agent_name}")

        # Use fallback prompt
        fallback_key = self.prompt_id.split('/')[-1]
        return FALLBACK_PROMPTS.get(fallback_key, "You are a helpful AI assistant.")

    def execute(self, input_data: str, context: Dict[str, Any] = None) -> str:
        """Execute the agent with given input"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_data}
            ]

            # Add context if provided
            if context:
                context_str = f"\n\nContext: {context}"
                messages[1]["content"] += context_str

            # Invoke LLM with Langfuse callbacks if available
            if self.langfuse_handler:
                config = RunnableConfig(
                    callbacks=[self.langfuse_handler],
                    tags=[self.agent_name, self.prompt_id.split('/')[-1]],
                    metadata={
                        "agent_type": self.agent_name,
                        "prompt_id": self.prompt_id
                    }
                )
                response = self.llm.invoke(messages, config=config)
            else:
                response = self.llm.invoke(messages)

            logger.info(f"{self.agent_name} executed successfully")
            return response.content

        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            return f"Error in {self.agent_name}: {str(e)}"


class TopicExtractionAgent(SubAgent):
    """Agent: Extracts topics from candidate answers"""

    def __init__(self, llm: ChatOpenAI, langfuse_client: Optional[Langfuse] = None, langfuse_handler: Optional[CallbackHandler] = None, prompt_label: str = "production"):
        super().__init__(
            agent_name="topic_extraction",
            prompt_id="AIInterviewer/eval/topic_extraction",
            llm=llm,
            langfuse_client=langfuse_client,
            langfuse_handler=langfuse_handler,
            prompt_label=prompt_label
        )


class IdealGenerationAgent(SubAgent):
    """Agent: Generates ideal/reference answers"""

    def __init__(self, llm: ChatOpenAI, langfuse_client: Optional[Langfuse] = None, langfuse_handler: Optional[CallbackHandler] = None, prompt_label: str = "production"):
        super().__init__(
            agent_name="ideal_generation",
            prompt_id="AIInterviewer/eval/ideal_generation",
            llm=llm,
            langfuse_client=langfuse_client,
            langfuse_handler=langfuse_handler,
            prompt_label=prompt_label
        )


class TopicEvaluatorAgent(SubAgent):
    """Agent: Evaluates answers by topic"""

    def __init__(self, llm: ChatOpenAI, langfuse_client: Optional[Langfuse] = None, langfuse_handler: Optional[CallbackHandler] = None, prompt_label: str = "production"):
        super().__init__(
            agent_name="topic_evaluator",
            prompt_id="AIInterviewer/eval/topic_evaluator",
            llm=llm,
            langfuse_client=langfuse_client,
            langfuse_handler=langfuse_handler,
            prompt_label=prompt_label
        )


class CrossValidationAgent(SubAgent):
    """Agent: Cross-validates and provides final evaluation"""

    def __init__(self, llm: ChatOpenAI, langfuse_client: Optional[Langfuse] = None, langfuse_handler: Optional[CallbackHandler] = None, prompt_label: str = "production"):
        super().__init__(
            agent_name="cross_validation",
            prompt_id="AIInterviewer/eval/cross_validation",
            llm=llm,
            langfuse_client=langfuse_client,
            langfuse_handler=langfuse_handler,
            prompt_label=prompt_label
        )


class EvaluationPipeline:
    """Orchestrates the Evaluation pipeline (4 agents)"""

    def __init__(self, llm: ChatOpenAI, langfuse_client: Optional[Langfuse] = None, langfuse_handler: Optional[CallbackHandler] = None, prompt_label: str = "production"):
        self.topic_extraction = TopicExtractionAgent(llm, langfuse_client, langfuse_handler, prompt_label)
        self.ideal_generation = IdealGenerationAgent(llm, langfuse_client, langfuse_handler, prompt_label)
        self.topic_evaluator = TopicEvaluatorAgent(llm, langfuse_client, langfuse_handler, prompt_label)
        self.cross_validation = CrossValidationAgent(llm, langfuse_client, langfuse_handler, prompt_label)
        logger.info("Evaluation Pipeline initialized")

    def execute(self, user_input: str) -> Dict[str, Any]:
        """Execute the full Evaluation pipeline"""
        logger.info("Starting Evaluation Pipeline")

        # Step 1: Extract topics from answer
        topics = self.topic_extraction.execute(user_input)

        # Step 2: Generate ideal answer
        ideal_answer = self.ideal_generation.execute(
            user_input,
            context={"extracted_topics": topics}
        )

        # Step 3: Evaluate by topic
        topic_scores = self.topic_evaluator.execute(
            user_input,
            context={
                "extracted_topics": topics,
                "ideal_answer": ideal_answer
            }
        )

        # Step 4: Cross-validate and finalize
        final_evaluation = self.cross_validation.execute(
            user_input,
            context={
                "extracted_topics": topics,
                "ideal_answer": ideal_answer,
                "topic_scores": topic_scores
            }
        )

        logger.info("Evaluation Pipeline completed")
        return {
            "pipeline": "EVAL_PIPELINE",
            "extracted_topics": topics,
            "ideal_answer": ideal_answer,
            "topic_scores": topic_scores,
            "final_evaluation": final_evaluation,
            "final_output": final_evaluation
        }


# ==========================================
# SUPERVISOR TOOL WRAPPERS
# ==========================================

def create_supervisor_tools(
    qg_agent,
    eval_pipeline: EvaluationPipeline
) -> List[Callable]:
    """
    Create high-level tools for the supervisor agent.

    The supervisor sees 2 main capabilities:
    1. generate_questions_and_answers - Calls the Question & Answer Generator Agent
    2. evaluate_answer - Calls the Evaluation Pipeline

    Args:
        qg_agent: The Question & Answer Generator Agent (LangChain agent)
        eval_pipeline: The Evaluation Pipeline instance

    Returns:
        List of tool functions for the supervisor
    """

    @tool
    def generate_questions_and_answers(request: str) -> str:
        """
        Generate interview questions AND ideal answers based on candidate profile.

        Use this tool when the user wants to:
        - Generate interview questions
        - Create assessments
        - Develop interview material
        - Get questions for a specific role/candidate

        This tool calls an INTELLIGENT AGENT that:
        1. Analyzes candidate resume vs job requirements
        2. Searches knowledge base for relevant questions
        3. Researches current industry trends
        4. Generates appropriate questions with ideal answers

        Args:
            request: Can be either:
                - Simple text like "Generate Python questions"
                - JSON with resume, job_profile, and num_questions

        Returns:
            JSON with questions and ideal answers
        """
        try:
            logger.info(f"Tool 'generate_questions_and_answers' invoked")

            # Invoke the Question & Answer Generator Agent
            result = qg_agent.invoke({
                "messages": [{"role": "user", "content": request}]
            })

            # Extract final message
            final_message = result["messages"][-1] if result.get("messages") else None
            final_output = final_message.content if final_message else "No questions generated"

            logger.info("Question & Answer generation completed")
            return final_output

        except Exception as e:
            logger.error(f"Error in generate_questions_and_answers tool: {e}")
            return f"Error generating questions and answers: {str(e)}"

    @tool
    def evaluate_answer(request: str) -> str:
        """
        Evaluate candidate's answer to an interview question.

        Use this tool when the user wants to:
        - Evaluate candidate answers
        - Score responses
        - Get feedback on answers
        - Check answer correctness
        - Assess interview performance

        Args:
            request: The question and candidate's answer to evaluate
                    (e.g., "Question: What is polymorphism? Answer: Polymorphism is...")

        Returns:
            Detailed evaluation with scores and feedback
        """
        try:
            logger.info(f"Tool 'evaluate_answer' invoked")
            result = eval_pipeline.execute(request)
            return result.get("final_output", "No evaluation generated")
        except Exception as e:
            logger.error(f"Error in evaluate_answer tool: {e}")
            return f"Error evaluating answer: {str(e)}"

    return [generate_questions_and_answers, evaluate_answer]
