#!/usr/bin/env python3
"""
AI Interviewer CLI

Interactive command-line interface for the AI Interviewer system.
Supports both Question Generation and Evaluation pipelines.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_interviewer.agents.agent_service import AgentService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InterviewCLI:
    """Interactive CLI for AI Interviewer"""

    def __init__(self):
        """Initialize the CLI"""
        load_dotenv()
        self.agent_service: Optional[AgentService] = None
        self.session_id = None

    def display_banner(self):
        """Display welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║            🎯 AI INTERVIEWER SYSTEM 🎯                        ║
║                                                               ║
║  Powered by LangGraph + Langfuse                              ║
║                                                               ║
║  Features:                                                    ║
║  • Question Generation Pipeline (4 agents)                    ║
║  • Evaluation Pipeline (4 agents)                             ║
║  • Intelligent LLM-based routing                              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
        print(banner)

    def display_help(self):
        """Display help information"""
        help_text = """
📚 AVAILABLE COMMANDS:

  /help              - Show this help message
  /quit or /exit     - Exit the application
  /clear             - Clear the screen
  /session <id>      - Set session ID for tracking
  /info              - Show current configuration
  /examples          - Show example queries

💡 USAGE EXAMPLES:

  Question Generation:
    • "Generate 5 Python interview questions for a senior developer"
    • "Create MCQs on machine learning algorithms"
    • "I need questions about React hooks for a mid-level candidate"

  Answer Evaluation:
    • "Evaluate this answer: Q: What is Python? A: Python is a programming language"
    • "Score the candidate's response to the OOP question"
    • "Check if this answer about SQL joins is correct"

Simply type your request and press Enter!
"""
        print(help_text)

    def display_examples(self):
        """Display example queries"""
        examples = """
📝 EXAMPLE QUERIES:

🔹 Question Generation:
   1. "Generate 3 MCQs on Python data structures"
   2. "Create interview questions for a senior DevOps engineer"
   3. "I need behavioral questions for a team lead position"
   4. "Generate algorithm questions with medium difficulty"

🔹 Evaluation:
   1. "Evaluate: Q: What is Git? A: Git is a version control system used for tracking changes"
   2. "Score this SQL answer: SELECT * FROM users WHERE age > 18"
   3. "Check this JavaScript answer about closures: [candidate's answer]"
   4. "Evaluate the candidate's response about microservices architecture"

Type any of these to try them out!
"""
        print(examples)

    def display_info(self):
        """Display current configuration"""
        info = f"""
ℹ️  CURRENT CONFIGURATION:

  • OpenAI Model: {os.getenv('OPENAI_MODEL', 'gpt-4o')}
  • Langfuse Enabled: {'Yes' if os.getenv('LANGFUSE_PUBLIC_KEY') else 'No'}
  • Session ID: {self.session_id or 'Not set'}
  • Status: {'Connected' if self.agent_service else 'Not initialized'}

Environment variables loaded from .env file.
"""
        print(info)

    def initialize_service(self):
        """Initialize the agent service"""
        try:
            print("\n🔄 Initializing AI Interviewer Service...")
            print("   This may take a few seconds...\n")

            self.agent_service = AgentService()

            print("✅ Service initialized successfully!")
            print("   All agents are ready.\n")
            return True

        except Exception as e:
            print(f"\n❌ Failed to initialize service: {e}")
            print("\nPlease check:")
            print("  1. Your .env file contains valid API keys")
            print("  2. OPENAI_API_KEY is set correctly")
            print("  3. Internet connection is available\n")
            return False

    def process_command(self, user_input: str) -> bool:
        """
        Process special commands.

        Returns:
            True if should continue, False if should exit
        """
        user_input = user_input.strip()

        if user_input in ['/quit', '/exit']:
            return False

        elif user_input == '/help':
            self.display_help()

        elif user_input == '/clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            self.display_banner()

        elif user_input.startswith('/session '):
            session_id = user_input.split(' ', 1)[1].strip()
            self.session_id = session_id
            print(f"✅ Session ID set to: {session_id}\n")

        elif user_input == '/info':
            self.display_info()

        elif user_input == '/examples':
            self.display_examples()

        else:
            # Not a command, process as normal input
            return None

        return True

    def run_query(self, user_input: str):
        """Run a user query through the agent service"""
        try:
            print("\n🤔 Processing your request...")
            print("   Supervisor is analyzing and routing...\n")

            # Run the agent service
            result = self.agent_service.run(
                user_input=user_input,
                session_id=self.session_id
            )

            # Display results
            print("═" * 70)
            print(f"✅ Status: Request processed successfully")

            # Show routing decision
            if result.get('pipeline'):
                print(f"🎯 Supervisor Decision: {result.get('pipeline')}")
                print(f"🔧 Tool Used: {result.get('tool_used', 'Unknown')}")

            print(f"🤖 Model: {result.get('metadata', {}).get('model', 'gpt-4o')}")
            if self.session_id:
                print(f"📋 Session ID: {self.session_id}")
            print("═" * 70)

            if result.get('error'):
                print(f"\n⚠️  ERROR: {result['error']}\n")

            print("\n📝 RESULT:\n")
            print(result.get('result', 'No result generated'))
            print("\n" + "═" * 70 + "\n")

        except Exception as e:
            print(f"\n❌ Error processing query: {e}\n")
            logger.error(f"Query processing error: {e}", exc_info=True)

    def run(self):
        """Main CLI loop"""
        self.display_banner()

        # Initialize service
        if not self.initialize_service():
            print("Exiting due to initialization failure.")
            return

        print("Type '/help' for available commands or start asking questions!\n")

        # Main interaction loop
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("👤 You: ").strip()

                    if not user_input:
                        continue

                    # Process commands
                    command_result = self.process_command(user_input)

                    if command_result is False:
                        # Exit requested
                        break
                    elif command_result is True:
                        # Command processed, continue
                        continue
                    # else: Not a command, process as query

                    # Run query
                    self.run_query(user_input)

                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted by user")
                    break

                except EOFError:
                    print("\n\n👋 End of input detected")
                    break

        finally:
            # Cleanup
            print("\n🔄 Shutting down...")
            if self.agent_service:
                self.agent_service.shutdown()
            print("👋 Thank you for using AI Interviewer System!")
            print("   Check your traces at: https://cloud.langfuse.com\n")


def main():
    """Entry point"""
    cli = InterviewCLI()
    cli.run()


if __name__ == "__main__":
    main()
