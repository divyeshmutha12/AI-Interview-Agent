"""
Supervisor Agent for AI Interviewer System

This agent uses the LangChain supervisor pattern with high-level tools.
The supervisor coordinates between question generation and evaluation pipelines
using tool-based routing instead of manual decision logic.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent  # Built-in langchain 1.0+ function
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any, Optional, List, Callable
import logging
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

# Fallback supervisor prompt (used if Langfuse is unavailable)
FALLBACK_SUPERVISOR_PROMPT = """
You are a Supervisor Agent for an AI Interviewer System.

You have access to the following tools:
1. generate_questions_and_answers - For generating interview questions with ideal answers
2. evaluate_answer - For evaluating candidate answers

IMPORTANT: You MUST use the appropriate tool to complete the user's request. Never just describe what you would do - actually call the tool!

## Input Formats:

### For Question Generation:
The user will provide:
- Candidate Resume (full text or structured data)
- Job Description/Profile (full text or requirements)
- Number of Questions (optional, default: 5)

You should extract this information and format it as a structured request to pass to generate_questions_and_answers tool.

Example Input:
```
Resume: [candidate resume text]
Job Profile: [job description text]
Number of Questions: 3
```

You should call generate_questions_and_answers with:
```
Generate 3 interview questions for this candidate.

Resume:
[full resume text]

Job Profile:
[full job description]

Number of Questions: 3
```

### For Answer Evaluation:
The user will provide:
- Question asked
- Candidate's answer

Example Input:
```
Question: What is polymorphism?
Answer: Polymorphism allows objects of different types to be treated as objects of a common type...
```

## Workflow:

1. **Identify the task**: Question generation or answer evaluation
2. **Extract information**: Resume, JD, questions OR question and answer
3. **Format the request**: Structure it properly for the tool
4. **Call the appropriate tool**: Pass the formatted request
5. **Present results**: Show the tool's output to the user

## Examples:

**Example 1: Question Generation**
User: "Generate 5 Python questions for a senior developer with 6 years experience in Django and PostgreSQL"
Action: Call generate_questions_and_answers with the request

**Example 2: With Full Resume**
User provides resume text, job description, and number of questions
Action: Extract and format all information, then call generate_questions_and_answers

**Example 3: Answer Evaluation**
User: "Evaluate: Q: What is Git? A: Version control system for tracking changes"
Action: Call evaluate_answer with the question and answer

ALWAYS call the appropriate tool - do not just respond with text!
"""


def create_supervisor(
    llm: ChatOpenAI,
    tools: List[Callable],
    langfuse_client: Optional[Langfuse] = None,
    langfuse_handler: Optional[CallbackHandler] = None,
    prompt_label: str = "production"
):
    """
    Create a supervisor agent using the LangChain pattern.

    The supervisor coordinates between high-level pipeline tools,
    making intelligent routing decisions based on user requests.

    Args:
        llm: The LLM model for the supervisor
        tools: List of tool functions (wrapped pipelines)
        langfuse_client: Optional Langfuse client for prompt management
        langfuse_handler: Optional Langfuse callback handler
        prompt_label: Label for Langfuse prompts (default: "production")

    Returns:
        Compiled StateGraph configured as a supervisor agent

    Example:
        >>> from .sub_agents import create_pipeline_tools
        >>> tools = create_pipeline_tools(qg_pipeline, eval_pipeline)
        >>> supervisor = create_supervisor(llm, tools=tools)
        >>> result = supervisor.invoke({"messages": [{"role": "user", "content": "Generate Python questions"}]})
    """
    logger.info("Creating supervisor agent with tool-based routing...")

    # Load system prompt from Langfuse or use fallback
    system_prompt = FALLBACK_SUPERVISOR_PROMPT

    # Try to load from Langfuse, but fallback on any error
    if langfuse_client:
        try:
            prompt_obj = langfuse_client.get_prompt(
                "AIInterviewer/supervisor",
                label=prompt_label
            )
            system_prompt = prompt_obj.compile()
            logger.info("Loaded supervisor prompt from Langfuse")
        except Exception as e:
            logger.warning(f"Failed to load Langfuse supervisor prompt: {e}")
            logger.info("Using fallback supervisor prompt")

    # Create supervisor agent using langchain's built-in create_agent
    # This returns a CompiledStateGraph that can be invoked
    supervisor_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    logger.info(f"Supervisor agent created with {len(tools)} tools")
    return supervisor_agent
