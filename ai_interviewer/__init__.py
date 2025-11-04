"""
AI Interviewer System

A comprehensive AI-powered interview system with:
- Question Generation Pipeline (4 agents)
- Evaluation Pipeline (4 agents)
- Supervisor Agent with LLM-based routing
- LangGraph orchestration
- Langfuse prompt management and tracing
"""

__version__ = "1.0.0"
__author__ = "AI Interviewer Team"

from .agents.agent_service import AgentService

__all__ = ["AgentService"]
