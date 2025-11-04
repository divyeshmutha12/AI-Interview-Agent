"""
Agent Service - Main orchestration using Supervisor Pattern

This service coordinates the Supervisor Agent and pipeline tools
following the LangChain supervisor pattern.

The supervisor uses high-level tools (generate_questions, evaluate_answer)
instead of manual routing logic.
"""

from langchain_openai import ChatOpenAI
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_core.runnables import RunnableConfig

from .supervisor_agent import create_supervisor
from .sub_agents import (
    create_question_answer_generator_agent,
    EvaluationPipeline,
    create_supervisor_tools
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentService:
    """
    Main Agent Service using Supervisor Pattern.

    Manages the supervisor agent which coordinates pipeline tools
    for question generation and answer evaluation.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        langfuse_public_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_host: Optional[str] = None,
        prompt_label: Optional[str] = None
    ):
        """
        Initialize the Agent Service.

        Args:
            openai_api_key: OpenAI API key
            openai_model: OpenAI model to use
            langfuse_public_key: Langfuse public key
            langfuse_secret_key: Langfuse secret key
            langfuse_host: Langfuse host URL
            prompt_label: Label for Langfuse prompts (developer name, environment, or status)
                         If None, reads from LANGFUSE_PROMPT_LABEL env var (default: "production")
        """
        logger.info("Initializing Agent Service with Supervisor Pattern...")

        # Load environment variables
        load_dotenv()

        # Setup OpenAI
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o")

        # Setup Prompt Label (allows per-developer or per-environment configuration)
        self.prompt_label = prompt_label or os.getenv("LANGFUSE_PROMPT_LABEL", "production")
        logger.info(f"Using prompt label: {self.prompt_label}")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")

        self.llm = ChatOpenAI(
            model=self.openai_model,
            api_key=self.openai_api_key,
            temperature=0.7
        )
        logger.info(f"OpenAI LLM initialized with model: {self.openai_model}")

        # Setup Langfuse
        self.langfuse_client = None
        self.langfuse_handler = None

        try:
            # Initialize Langfuse client for prompt management
            self.langfuse_client = Langfuse(
                public_key=langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
                host=langfuse_host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )

            # Initialize CallbackHandler for tracing
            # Note: In Langfuse v3, credentials are configured via environment variables
            # or through the Langfuse client. The CallbackHandler takes no constructor args.
            self.langfuse_handler = CallbackHandler()

            logger.info("Langfuse initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")
            logger.info("Continuing without Langfuse (using fallback prompts)")

        # Initialize agents and pipelines
        logger.info("Initializing Question & Answer Generator Agent...")
        self.qg_agent = create_question_answer_generator_agent(
            llm=self.llm,
            langfuse_client=self.langfuse_client,
            langfuse_handler=self.langfuse_handler,
            prompt_label=self.prompt_label
        )

        logger.info("Initializing Evaluation Pipeline...")
        self.eval_pipeline = EvaluationPipeline(
            llm=self.llm,
            langfuse_client=self.langfuse_client,
            langfuse_handler=self.langfuse_handler,
            prompt_label=self.prompt_label
        )

        # Create supervisor tools
        logger.info("Creating supervisor tools...")
        tools = create_supervisor_tools(self.qg_agent, self.eval_pipeline)

        # Create supervisor with tools
        logger.info("Creating supervisor agent...")
        self.supervisor = create_supervisor(
            llm=self.llm,
            tools=tools,
            langfuse_client=self.langfuse_client,
            langfuse_handler=self.langfuse_handler,
            prompt_label=self.prompt_label
        )

        logger.info("Agent Service initialized successfully")

    def run(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the agent service with user input.

        The supervisor agent will automatically route the request to the
        appropriate pipeline tool (generate_questions or evaluate_answer).

        Args:
            user_input: The user's query or request
            user_id: Optional user ID for tracking
            session_id: Optional session ID for grouping requests

        Returns:
            Dict containing the agent's response and metadata
        """
        try:
            logger.info(f"Running agent service for input: {user_input[:100]}...")

            # Prepare config with Langfuse callbacks for proper tracing
            invoke_config = None
            if self.langfuse_handler:
                # Build metadata with Langfuse-specific keys for tracing
                metadata = {
                    "agent_type": "supervisor"
                }

                # Add Langfuse-specific metadata for user/session tracking
                if user_id:
                    metadata["langfuse_user_id"] = user_id
                if session_id:
                    metadata["langfuse_session_id"] = session_id

                invoke_config = RunnableConfig(
                    callbacks=[self.langfuse_handler],
                    tags=["supervisor", "main_invocation"],
                    metadata=metadata
                )

            # Invoke supervisor with user input and Langfuse tracing
            # The supervisor will decide which tool(s) to use
            if invoke_config:
                result = self.supervisor.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=invoke_config
                )
            else:
                result = self.supervisor.invoke({
                    "messages": [{"role": "user", "content": user_input}]
                })

            # Extract the final output from agent response
            # The result contains the full message history
            final_message = result["messages"][-1] if result.get("messages") else None
            final_result = final_message.content if final_message else "No response generated"

            # Extract tool calls to determine which agent/pipeline was used
            tool_used = None
            pipeline_name = None
            for message in result.get("messages", []):
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_used = message.tool_calls[0]['name']
                    if tool_used == "generate_questions_and_answers":
                        pipeline_name = "Question & Answer Generation"
                    elif tool_used == "evaluate_answer":
                        pipeline_name = "Answer Evaluation"
                    break

            # Prepare output
            final_output = {
                "user_input": user_input,
                "result": final_result,
                "tool_used": tool_used,
                "pipeline": pipeline_name,
                "metadata": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "model": self.openai_model
                }
            }

            logger.info("Agent service execution completed successfully")
            return final_output

        except Exception as e:
            logger.error(f"Error running agent service: {e}")
            return {
                "user_input": user_input,
                "error": str(e),
                "result": f"System error: {str(e)}"
            }
        finally:
            # Flush Langfuse traces
            if self.langfuse_client:
                try:
                    self.langfuse_client.flush()
                except Exception as e:
                    logger.warning(f"Failed to flush Langfuse: {e}")

    def generate_interview_questions(
        self,
        candidate_resume: str,
        job_profile: str,
        num_questions: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate interview questions with ideal answers based on candidate and job.

        This is a convenience method that formats the inputs properly for the
        Question & Answer Generator Agent.

        Args:
            candidate_resume: Full text of candidate's resume/CV
            job_profile: Full text of job description/requirements
            num_questions: Number of questions to generate (default: 5)
            user_id: Optional user ID for tracking
            session_id: Optional session ID for grouping requests

        Returns:
            Dict containing:
                - result: Generated questions with ideal answers
                - pipeline: "Question & Answer Generation"
                - metadata: Generation metadata

        Example:
            >>> service = AgentService()
            >>> result = service.generate_interview_questions(
            ...     candidate_resume="John Doe, Senior Python Developer...",
            ...     job_profile="We are looking for a Senior Backend Engineer...",
            ...     num_questions=3
            ... )
            >>> print(result['result'])
        """
        # Format the request for the supervisor agent
        formatted_request = f"""
Generate {num_questions} interview questions for this candidate.

Resume:
{candidate_resume}

Job Profile:
{job_profile}

Number of Questions: {num_questions}

Please analyze the candidate's qualifications against the job requirements,
search for relevant topics, and generate appropriate interview questions
with comprehensive ideal answers.
"""

        logger.info(f"Generating {num_questions} interview questions")

        # Use the standard run method
        return self.run(
            user_input=formatted_request,
            user_id=user_id,
            session_id=session_id
        )

    def chat(self, user_input: str, session_id: Optional[str] = None) -> str:
        """
        Simplified chat interface that returns just the result string.

        Args:
            user_input: The user's message
            session_id: Optional session ID

        Returns:
            The agent's response as a string
        """
        result = self.run(user_input, session_id=session_id)
        return result.get("result", "No response generated")

    def shutdown(self):
        """Cleanup and shutdown the service"""
        logger.info("Shutting down Agent Service...")
        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
                logger.info("Langfuse traces flushed")
            except Exception as e:
                logger.warning(f"Error flushing Langfuse: {e}")
        logger.info("Agent Service shutdown complete")
