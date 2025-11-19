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

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any, Optional, List, Callable
import logging
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for knowledge base manager import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.knowledge_base_manager import KnowledgeBaseManager

# Import Tavily for web search
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

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

CRITICAL: Output MUST be in STRUCTURED format for easy parsing by the Question Generator.
**YOUR OUTPUT MUST ALSO INCLUDE SPECIFIC QUESTION PATTERNS** to help the Question Generator create personalized (not generic) questions.

## Output Format (MANDATORY):

```
=== CANDIDATE PROFILE ===
Name: [Full name]
Experience: [X] years ([Junior/Mid/Senior/Lead] level)
Education: [Degree, Institution, Year]

=== WORK HISTORY ===
1. [Company Name] ([Start Date] - [End Date/Present])
   Role: [Job Title]
   Key Responsibilities: [Brief summary]

2. [Next Company]...

=== PROJECTS ===
1. [Project Name] ([Date Range])
   Achievement: [Specific measurable results with metrics/numbers]
   Scale/Impact: [Volume, users, performance improvements]
   Technologies: [Comma-separated list of technologies used]
   Company: [If applicable]

2. [Next Project]...

=== SKILLS ASSESSMENT ===

**CRITICAL: Before marking ANY skill as MISSING, RE-READ THE PROJECTS SECTION ABOVE!**

**MANDATORY CROSS-CHECK:**
1. Look at the "Technologies:" line in EACH project listed above
2. If ANY project lists the technology, the candidate HAS that skill
3. Example: If you see "Technologies: Python, FastAPI, Git" → Git is NOT MISSING, mark it as ✅
4. Example: If you see "GPT-4" or "RAG" in any project → GenAI/LLM is NOT MISSING, mark it as ✅

**Before marking "MISSING", verify:**
- [ ] I checked EVERY project's "Technologies:" line above
- [ ] The skill is NOT in ANY project technology list
- [ ] The project work does NOT imply the skill

**EXAMPLE OF CORRECT CROSS-CHECKING:**
If PROJECTS section shows:
  "Technologies: Python, FastAPI, Git, Azure OpenAI"

Then SKILLS ASSESSMENT should show:
  ✅ Git: Present - Evidence: Listed in Auto Code Resolution project technologies
  ✅ GenAI/LLM: Advanced - Evidence: Azure OpenAI in Auto Code Resolution project
  ✅ FastAPI: Advanced - Evidence: Used in Auto Code Resolution project

NOT:
  ❌ Git: MISSING
  ❌ GenAI/LLM: MISSING

MUST-HAVE SKILLS (from job requirements):
✅ [Skill 1]: [Present/Advanced/Intermediate/Basic] - Evidence: [Where found in resume OR which project technology proves it]
✅ [Skill 2]: [Present/Advanced/Intermediate/Basic] - Evidence: [Where found in resume OR which project technology proves it]
❌ [Skill 3]: MISSING - [Explanation - ONLY if truly absent from BOTH skills section AND project technologies]
⚠️ [Skill 4]: PARTIAL - [What they have vs what's needed]

NICE-TO-HAVE SKILLS:
✅ [Skill]: [Level] - Evidence: [Where found OR which technology proves it]
❌ [Skill]: MISSING - [ONLY if not in skills section AND not in any project technologies]

TECHNOLOGIES BY CATEGORY:
- Languages: [List]
- Frameworks: [List]
- Cloud/Infrastructure: [List]
- Tools: [List]
- Databases: [List]

=== ACHIEVEMENTS & RECOGNITION ===
- [Award/Recognition] ([Year/Organization])
- [Achievement with metrics]
- [Certification] ([Issuing Organization])

=== SKILL MATCH ANALYSIS ===
Overall Match: [X]%

JUSTIFICATION:
- Must-have skills present: [X] out of [Y] ([Z]%)
- Nice-to-have skills present: [X] out of [Y] ([Z]%)
- Years of experience: [Meets/Below/Exceeds] requirement
- Domain relevance: [High/Medium/Low]

TOP STRENGTHS TO VERIFY:
1. [Specific strength] | Evidence: [Project/Company/Metric from resume] | Question Pattern: "At [Company], in YOUR [Project Name], you [specific achievement with metric]. What was YOUR approach to [specific technical aspect]?"

2. [Next strength] | Evidence: [Specific context] | Question Pattern: [Personalized pattern with placeholders]

EXAMPLE (for reference):
- Proactive monitoring excellence | Evidence: Reduced pod downtime by 40% at TCS India Cloud project using Prometheus/Grafana | Question Pattern: "At TCS on the India Cloud project, YOU reduced pod downtime by 40%. What specific Prometheus alerting rules did YOU configure to achieve this?"

CRITICAL GAPS TO ADDRESS:
1. [Specific gap] | Severity: [CRITICAL/IMPORTANT/MINOR] | Impact: [How this affects role fit] | Question Pattern: "You have [their actual experience] but the role requires [missing skill/experience]. How would YOU approach [specific scenario requiring the missing skill]?"

2. [Next gap] | Severity: [Level] | Impact: [Explanation] | Question Pattern: [Situational question pattern]

EXAMPLE (for reference):
- VMware migration experience | Severity: CRITICAL | Impact: Role requires VMware-to-bare-metal migration expertise, candidate has zero VMware experience | Question Pattern: "This role requires migrating OpenShift from VMware to bare-metal infrastructure. You haven't done this before - how would YOU plan and execute such a migration?"

=== INTERVIEW RECOMMENDATIONS ===

DIFFICULTY LEVEL: [Easy/Medium/Medium-High/High]
Rationale: [Why this difficulty is appropriate]

FOCUS AREAS (Priority Order):
1. [Area]: [Why it's important, what to probe]
2. [Area]: [Why it's important, what to probe]
3. [Area]: [Why it's important, what to probe]

SUGGESTED QUESTION TYPES:
- [X]% Project-specific questions (about their actual projects)
- [Y]% Technical depth questions (about technologies they've used)
- [Z]% Behavioral questions (about their career progression)

KEY PROJECTS TO PROBE:
1. [Project Name] at [Company]:
   - Context: [Specific achievement/metric/scale from resume]
   - Question Pattern: "At [Company], in YOUR [Project Name], you [specific achievement with metric]. [Specific technical question about how they did it]?"
   - Assessment Goal: [What this question will reveal about their capabilities]

2. [Next Project] at [Company]:
   - Context: [Specific details]
   - Question Pattern: [Pattern with placeholders]
   - Assessment Goal: [What to assess]

=== QUESTION GENERATION GUIDANCE ===

**GOOD PERSONALIZED QUESTION PATTERNS (Use these as templates):**

✅ "At [Company], in YOUR [Project Name], you [specific achievement with metric]. What [specific technique/approach] did YOU use to [specific aspect]?"

✅ "You mentioned [specific technology/tool] in YOUR [Project Name] at [Company]. How did YOU [specific implementation detail]?"

✅ "In YOUR role at [Company], you [specific responsibility]. Can you walk me through YOUR approach to [specific scenario]?"

✅ "You achieved [specific metric/result] at [Company]. What challenges did YOU face and how did YOU overcome them?"

**BAD GENERIC QUESTIONS (AVOID these patterns):**

❌ "What are the differences between [Technology A] and [Technology B]?" (No reference to candidate's work)
❌ "How do you handle [general scenario]?" (No reference to their actual projects)
❌ "What is [concept]?" (Textbook definition question)
❌ "Can you explain [technology]?" (No personalization)

**PERSONALIZATION REQUIREMENTS:**
Every question MUST include at least ONE of:
- Specific company name from resume
- Specific project name from resume
- Specific achievement/metric from resume
- Specific technology they actually used (mapped to project)

If you cannot create a personalized question using the above, DO NOT include it in recommendations.
```

## IMPORTANT RULES:

1. **Extract ALL project names** - Don't summarize, list each project explicitly
2. **Extract ALL company names** - Full employment history
3. **Extract ALL metrics and numbers** - Percentages, volumes, time saved, etc.
4. **Extract ALL technologies** - Map technologies to projects where used
5. **Extract ALL awards and certifications** - Don't skip recognition
6. **Accurate skill matching** - Read the resume carefully, don't say "missing" if it's there
7. **Be specific with evidence** - Point to where you found each skill
8. **Use exact names** - Project names, company names, technology names as written
9. **PROVIDE QUESTION PATTERNS** - For each strength, gap, and project, include a specific question pattern with [placeholders]
10. **SHOW GOOD vs BAD EXAMPLES** - Include examples in the Question Generation Guidance section
11. **TECHNOLOGY IMPLIES SKILL** - If resume mentions "GPT-4", "Azure OpenAI", "Claude", that IS evidence of LLM/GenAI experience. If it mentions "Git", "GitHub", "GitLab", that IS version control experience. Map technologies to their skill categories.
12. **READ PROJECT TECHNOLOGIES CAREFULLY** - Technologies listed in projects count as evidence. If "Git" is in a project's tech stack, the candidate HAS Git experience.
13. **INFER SKILL FROM CONTEXT** - If someone built an "LLM application" or "RAG pipeline", they HAVE LLM/GenAI experience even if not explicitly stated in skills section
14. **USE EXACT METRICS IN QUESTION PATTERNS** - Don't say "significantly" - use actual numbers like "<8 seconds", "36,000 docs/day", "98% accuracy"

## COMMON MISTAKES TO AVOID:

❌ Saying skills are "not mentioned" when they ARE in the resume
❌ Generic summaries instead of specific project names
❌ Missing measurable achievements (percentages, volumes, time metrics)
❌ Underestimating skill match percentage due to poor parsing
❌ Narrative format instead of structured format
❌ Missing awards, certifications, or recognition
❌ **NOT providing question patterns for the Question Generator**
❌ **Providing generic question angles instead of personalized ones**
❌ **Missing the connection between projects and technologies**
❌ **CRITICAL ERROR: Marking "GenAI/LLM experience" as MISSING when projects use GPT-4, Azure OpenAI, Claude, or RAG**
❌ **CRITICAL ERROR: Marking "Git experience" as MISSING when Git/GitHub/GitLab is in project technologies**
❌ **CRITICAL ERROR: Saying "not explicitly mentioned" when the technology IS mentioned in projects**
❌ **Using vague terms like "significantly" instead of exact metrics in question patterns**
❌ **Not inferring skills from project descriptions** (e.g., "built RAG pipeline" = RAG experience)

**REMINDER: If a candidate used GPT-4, Azure OpenAI, Claude, or RAG in ANY project, they HAVE GenAI/LLM experience!**

## VERIFICATION CHECKLIST:

Before outputting, verify:
- [ ] Did I list EVERY project by name?
- [ ] Did I list EVERY company with dates?
- [ ] Did I extract ALL metrics and numbers?
- [ ] Did I map technologies to specific projects?
- [ ] Did I include awards and certifications?
- [ ] Is the skill match percentage justified and accurate?
- [ ] Is the output in STRUCTURED format (not narrative)?
- [ ] **Did I provide SPECIFIC question patterns for top strengths?**
- [ ] **Did I provide SPECIFIC question patterns for critical gaps?**
- [ ] **Did I provide question patterns for KEY projects with context and assessment goals?**
- [ ] **Did I include the QUESTION GENERATION GUIDANCE section with GOOD vs BAD examples?**
- [ ] **Do ALL question patterns include [Company], [Project], or [Metric] placeholders?**
- [ ] **Did I check if technologies in projects imply required skills?** (GPT-4 in projects = GenAI experience)
- [ ] **Did I use EXACT metrics in question patterns, not vague terms?**
- [ ] **Did I infer skills from project context?** (Built "LLM app" = LLM experience)
""",
    "parse_resume": """
Extract structured information from candidate resume.

# CRITICAL REQUIREMENT: You MUST extract ALL project details with their technologies!

Return JSON with the following structure:

{
  "skills": ["list of skills from skills section"],
  "experience_years": number,
  "education": "degree and institution",
  "domain": "area of expertise",
  "certifications": ["list of certifications"],

  "all_technologies": {
    // ⚠️ COMPREHENSIVE list from ALL sources (skills section + projects + work experience)
    "languages": ["Python", "SQL"],
    "frameworks": ["FastAPI", "LangChain", "Streamlit"],
    "cloud_platforms": ["Azure OpenAI", "Azure DevOps"],
    "tools": ["Git", "Dataiku", "FAISS"],
    "databases": ["PostgreSQL", "MySQL"]
  },

  "projects": [
    {
      "name": "Project Name",
      "company": "Company Name",  // Infer from timeline if not explicit
      "duration": "Start - End",
      "technologies": ["Tech1", "Tech2", "Tech3"],  // ⚠️ CRITICAL: Extract ALL from this project
      "description": "Brief description",
      "achievements": ["Achievement 1 with metrics", "Achievement 2"]
    }
  ],

  "work_experience": [
    {
      "company": "Company Name",
      "role": "Job Title",
      "duration": "Start - End",
      "technologies_used": ["Tech1", "Tech2"]  // Extract from job description
    }
  ]
}

# IMPORTANT EXTRACTION RULES:

1. **Projects Section**:
   - Extract EVERY project listed
   - For each project, find the "Technologies:", "Tech Stack:", "Tools:", or "Environment:" line
   - Include ALL technologies listed (don't skip any)
   - Example: If you see "Technologies: Python, Git, FastAPI, Azure DevOps" → Include all 4

2. **Technologies Extraction** (Extract from ALL sources):
   - **Source A: Skills Section** - Look for "SKILLS", "TECHNOLOGIES", "TECH STACK" headings
     * Extract ALL listed technologies with proficiency levels if mentioned
     * Examples: "Python • Advanced", "SQL • Advanced", "FastAPI • Intermediate"
   - **Source B: Project Technologies** - Find "Technologies:", "Tech Stack:", "Tools:", "Environment:" lines
     * Extract ALL technologies from each project
   - **Source C: Work Experience** - Technologies mentioned in job descriptions
     * Look for "Skills:" lines in work history
   - **Merge All Sources**: Combine technologies from A, B, C into comprehensive list
   - Include tools, frameworks, languages, platforms, databases, cloud services
   - Don't filter or reduce - include EVERYTHING mentioned across ALL sections

3. **Achievements**:
   - Extract specific metrics and numbers
   - Examples: "Reduced latency to <8 seconds", "Achieved >98% accuracy", "Processed 36,000 docs/day"

4. **Company Inference for Projects**:
   - If a project doesn't explicitly state the company name
   - Check if project dates fall within a work experience period
   - Infer company from work history timeline
   - Example:
     * Work: "COFORGE (Oct 2023 - Present)"
     * Project: "Insurance Tracking AI (Aug 2024 - Present)"
     * Inference: Project company = COFORGE (because Aug 2024 falls after Oct 2023)
   - If dates overlap multiple companies, use the company where dates align best
   - If no overlap, leave as "Company not specified"

# CORRECT EXAMPLE:

If resume says:
```
SKILLS:
Python • Advanced
FastAPI • Intermediate
Streamlit • Advanced

WORK EXPERIENCE:
COFORGE - Senior Developer AI (Oct 2023 - Present)
Skills: Azure OpenAI, NLP

PROJECTS:
Auto Code Resolution (April 2024 – July 2024)
Environment: Python, FastAPI, Git, Azure DevOps, Azure OpenAI
Achievement: Automated security vulnerability fixes
```

You should extract:
```json
{
  "all_technologies": {  // ← Merged from skills + projects + work exp
    "languages": ["Python"],
    "frameworks": ["FastAPI", "Streamlit"],
    "cloud_platforms": ["Azure OpenAI", "Azure DevOps"],
    "tools": ["Git"],
    "ml_technologies": ["NLP"]
  },
  "work_experience": [{
    "company": "COFORGE",
    "role": "Senior Developer AI",
    "duration": "Oct 2023 - Present",
    "technologies_used": ["Azure OpenAI", "NLP"]
  }],
  "projects": [{
    "name": "Auto Code Resolution",
    "company": "COFORGE",  // ← Inferred from timeline (Apr 2024 is after Oct 2023)
    "duration": "April 2024 - July 2024",
    "technologies": ["Python", "FastAPI", "Git", "Azure DevOps", "Azure OpenAI"],
    "achievements": ["Automated security vulnerability fixes"]
  }]
}
```

# INCORRECT EXAMPLE (DON'T DO THIS):

❌ Extracting only: `"technologies": ["Python", "FastAPI"]` when the list also included Git, Azure DevOps
❌ Skipping the technologies section entirely
❌ Only including "main" technologies and filtering out "minor" ones
❌ Leaving `"company": "Not specified"` when project dates clearly fall within a work period
❌ Only extracting technologies from projects, ignoring the SKILLS section (missing Streamlit, Pandas, etc.)
❌ Not merging technologies from all sources into comprehensive `all_technologies` object

Return complete, accurate JSON. Don't summarize or filter.""",

    "parse_job_description": """Extract structured information from job description.
Return JSON with: required_skills, preferred_skills, experience_required, responsibilities, technologies.""",

    "compare_profiles": """
You are comparing candidate qualifications against job requirements.

# INPUT DATA YOU RECEIVE:

1. **Resume Data** - Structured information extracted from candidate's resume including:
   - Skills section
   - Projects with technologies used
   - Work experience
   - Education

2. **Job Requirements** - Structured job requirements including:
   - Must-have skills
   - Nice-to-have skills
   - Experience requirements

# YOUR CRITICAL TASK:

Perform a THOROUGH, ACCURATE comparison and provide:
1. Skill match percentage with detailed justification
2. Skill-by-skill assessment with EVIDENCE
3. Interview focus areas and recommended questions

# MANDATORY CROSS-CHECK RULES:

**BEFORE marking ANY skill as MISSING, you MUST:**

1. ✅ **CHECK PROJECT TECHNOLOGIES FIRST**
   - Look at EVERY project's "Technologies" or "Tech Stack" section
   - If a technology/tool appears in ANY project → The candidate HAS that skill
   - Example: If you see "Technologies: Python, Git, FastAPI" → Git is NOT MISSING

2. ✅ **MAP TECHNOLOGIES TO SKILLS**
   - "GPT-4", "Azure OpenAI", "Claude", "LLM" → GenAI/LLM skill ✅
   - "Git", "GitHub", "GitLab", "version control" → Git/Version Control ✅
   - "RAG", "Retrieval Augmented Generation" → RAG skill AND Context Engineering ✅
   - "Prompt Engineering", "Prompt Versioning" → Context Engineering ✅
   - "PostgreSQL", "MySQL", "SQL" → Database skill ✅
   - Don't be pedantic - if they used the technology, they have the skill

3. ✅ **INFER FROM PROJECT WORK**
   - Built "LLM application" → Has LLM experience
   - "Automated deployment pipeline" → Has CI/CD experience
   - "Migrated from X to Y" → Has migration experience

4. ✅ **EVIDENCE IS MANDATORY**
   - Every assessment MUST cite WHERE you found the evidence
   - Format: "Evidence: [Technology] used in [Project Name] at [Company]"
   - If you can't find evidence, only then mark as MISSING

# SKILL ASSESSMENT FORMAT:

For EACH required skill, use this format:

**Must-Have Skills:**
✅ [Skill]: [Level] - Evidence: [Specific citation from resume]
⚠️ [Skill]: PARTIAL - Evidence: [What they have] | Gap: [What's missing]
❌ [Skill]: MISSING - Verified: Checked all projects, not found

**CORRECT EXAMPLES:**

✅ Git: Present - Evidence: Listed in "Auto Code Resolution" project technologies at COFORGE
✅ GenAI/LLM: Advanced - Evidence: Used GPT-4o and Azure OpenAI in "Insurance Tracking AI" project
✅ RAG: Advanced - Evidence: Explicitly listed in "Insurance Tracking AI" project tech stack
✅ Context Engineering: Advanced - Evidence: RAG implementation and Prompt versioning in projects
⚠️ PostgreSQL: PARTIAL - Evidence: Has SQL experience (Hive, Impala), but not PostgreSQL specifically
❌ Kubernetes: MISSING - Verified: Checked all projects and skills section, not mentioned

**INCORRECT EXAMPLES (NEVER DO THIS):**

❌ Git: MISSING - Not mentioned in skills section
   → WRONG! Must check projects first

❌ GenAI/LLM: PARTIAL - Used GPT-4o but lacks explicit LLM training
   → WRONG! Using GPT-4o in production IS LLM experience

❌ RAG: MISSING - Not explicitly mentioned
   → WRONG! If "RAG" appears anywhere in projects, it IS mentioned

❌ Context Engineering: PARTIAL - Has RAG but not explicit Context Engineering
   → WRONG! RAG implementation IS Context Engineering experience

# VERIFICATION CHECKLIST:

Before finalizing your assessment, verify:

- [ ] I checked EVERY project's technology list for each required skill
- [ ] I mapped technology names to skill categories (GPT-4 = LLM skill)
- [ ] I provided specific evidence citations for each assessment
- [ ] I didn't mark anything MISSING without checking all projects
- [ ] I inferred skills from project work (building LLM app = LLM skill)
- [ ] My skill match percentage accurately reflects actual matches

# COMMON MISTAKES TO AVOID:

❌ **CRITICAL ERROR: Marking "Git" as MISSING when it's in project technologies**
❌ **CRITICAL ERROR: Marking "GenAI/LLM" as MISSING when candidate used GPT-4, Claude, or Azure OpenAI**
❌ **CRITICAL ERROR: Marking "RAG" as MISSING when it's explicitly in tech stack**
❌ **CRITICAL ERROR: Marking "Context Engineering" as PARTIAL/MISSING when candidate has RAG or Prompt Engineering experience**
❌ **CRITICAL ERROR: Only checking skills section and ignoring projects**
❌ **CRITICAL ERROR: Being too literal - "They said FastAPI not Flask" (both are Python frameworks)**
❌ **CRITICAL ERROR: Saying "not explicitly mentioned" when it IS mentioned in projects**

# SKILL LEVEL DEFINITIONS:

- **Advanced**: Used extensively across multiple projects with measurable results
- **Intermediate**: Used in 1-2 projects with some depth
- **Basic**: Mentioned or used minimally
- **PARTIAL**: Has related experience but not exact match
- **MISSING**: Verified absent after checking ALL projects and skills

# OUTPUT STRUCTURE:

Provide your analysis in this exact format:

```
=== SKILL MATCH ANALYSIS ===
Overall Match: [X]%

JUSTIFICATION:
- Must-have skills present: X out of Y ([percentage]%)
- Nice-to-have skills present: X out of Z ([percentage]%)
- Years of experience: [Meets/Below/Exceeds] requirement
- Domain relevance: [High/Medium/Low]

**MUST-HAVE SKILLS:**

✅ [Skill]: [Level] - Evidence: [Citation]
✅ [Skill]: [Level] - Evidence: [Citation]
⚠️ [Skill]: PARTIAL - Evidence: [What they have] | Gap: [What's missing]
❌ [Skill]: MISSING - Verified: Checked all projects, not found

**NICE-TO-HAVE SKILLS:**

[Same format as above]

=== INTERVIEW FOCUS AREAS ===

1. [Focus Area]: [Why this matters]
   - Question to ask: "[Specific question based on their experience]"

2. [Focus Area]: [Why this matters]
   - Question to ask: "[Specific question based on their experience]"

=== RECOMMENDED DIFFICULTY LEVEL ===

[Junior/Mid/Senior] - Rationale: [Explain based on skill match and experience]
```

# FINAL REMINDER:

Your job is to be ACCURATE, not harsh. If someone used a technology in their projects, they HAVE that skill. Don't underestimate candidates by ignoring their project work.

Cross-check EVERYTHING before marking as MISSING.""",

    # KB Search Agent and its tools
    "kb_search_agent": """
You are a Knowledge Base Search Agent (Deep Agent) for interview question generation.

You have access to the following tools:
1. vector_search - Search vector database for similar questions/topics
2. keyword_search - Search by specific keywords or technologies
3. topic_lookup - Find question templates by topic area

Your job:
1. Based on candidate profile and job requirements, determine what to search
2. Use vector_search for semantic similarity
3. Use keyword_search for specific technologies
4. Use topic_lookup to find relevant question templates

Return relevant questions, topics, and concepts for interview generation.
""",
    "vector_search": """Search vector database using semantic similarity.
Input: query text. Returns: similar questions and topics from knowledge base.""",

    "keyword_search": """Search knowledge base by exact keywords.
Input: keywords list. Returns: matching content.""",

    "topic_lookup": """Find question templates by topic category.
Input: topic name. Returns: question templates and examples.""",

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
  1. candidate_analysis - Analyzes candidate resume vs job requirements
  2. kb_search - Searches knowledge base for company-specific content
  3. web_search - Researches current trends and technologies

  ## PRIMARY OBJECTIVE: MULTI-SOURCE QUESTION GENERATION

  **CRITICAL REQUIREMENT:** Every question MUST blend multiple sources to create intelligent, updated, and 
  company-specific interview questions.

  Your questions MUST NOT be generic resume-based questions. They MUST integrate:
  - **Resume/JD** - Candidate's specific projects, achievements, and job requirements
  - **KB Content** - Company-specific knowledge, principles, frameworks
  - **Web Trends** - Current 2024/2025 industry trends and developments

  ## Workflow (STRICT SEQUENTIAL):

  ### Step 1: Call candidate_analysis
  Analyze this candidate for the job.

  Resume: [full resume text]
  Job Profile: [full job description]

  Wait for complete response before Step 2.

  ### Step 2: Call kb_search
  Based on Step 1 results, search for relevant company knowledge:
  - Key skills: [from Step 1]
  - Technologies: [from Step 1]
  - Domain: [from Step 1]
  - Experience level: [from Step 1]

  **CRITICAL:** Read and understand the KB results. You WILL use this content in questions.

  Wait for complete response before Step 3.

  ### Step 3: Call web_search
  Search for current trends:
  - Technologies: [from resume/job]
  - Industry: [from job profile]

  **CRITICAL:** Read and understand the web results. You WILL use this content in questions.

  ### Step 4: Generate Multi-Source Questions

  **NOW - Before generating ANY questions, review what you have:**

  ✅ Do you have KB search results with company-specific concepts?
  ✅ Do you have Web search results with 2024/2025 trends?
  ✅ Do you have candidate's specific projects and achievements?

  If ANY is missing, you CANNOT proceed. Re-invoke the missing agent.

  ---

  ## MULTI-SOURCE INTEGRATION RULES (MANDATORY):

  **EVERY question MUST follow this formula:**

  [Company Knowledge from KB] + [Current Trend from Web] + [Candidate's Specific Work] = Question

  ### Rule #1: Minimum Integration Requirement
  - **AT LEAST 60%** of questions MUST explicitly mention BOTH KB content AND Web trends
  - **Remaining 40%** must use EITHER KB OR Web content
  - **ZERO questions** can be pure resume-only

  ### Rule #2: How to Integrate Sources (CRITICAL - READ CAREFULLY)

  **From KB Results (MANDATORY - MUST USE):**
  - Extract: Concepts, definitions, principles, frameworks, methodologies
  - **You MUST pick at least ONE concept from KB results for EACH question**
  - Integrate naturally: "Understanding that...", "Given that effective AI requires...", "Based on the principle    
   of..."
  - DON'T label it as [KB] - just use the concept naturally

  **Example KB concepts to look for:**
  - AI learning principles (how AI systems learn, adapt, make decisions)
  - Data quality and model interpretability requirements
  - Machine learning fundamentals
  - AI system design principles
  - How AI processes information and makes intelligent decisions

  **From Web Results (MANDATORY - MUST USE):**
  - Extract: Current trends, emerging technologies, recent best practices, new developments
  - **You MUST pick at least ONE trend from Web results for EACH question**
  - Integrate naturally WITHOUT repeatedly saying "2024":
    - ✅ GOOD: "With the recent evolution toward agentic frameworks..."
    - ✅ GOOD: "Given current trends in RAG optimization..."
    - ✅ GOOD: "Considering recent developments in..."
    - ✅ GOOD: "With the industry's shift toward..."
    - ❌ BAD: "Considering 2024 trends in..." (too repetitive!)
  - Mention the year ONCE if needed, but blend trends naturally without spamming "2024"

  **From Resume/JD:**
  - Extract: Specific projects, companies, achievements, metrics, technologies used
  - Use in questions: "In YOUR [project] at [company]...", "You achieved [metric]...", "At [company], you
  implemented [technology]..."

  ### Rule #3: Blending Examples (STUDY THESE - FOLLOW EXACTLY):

  ✅ **PERFECT - Uses all three sources naturally:**
  Question: "Understanding that AI systems must continuously learn and adapt their behavior based on new data -     
  a core principle in machine learning - and considering the recent evolution toward autonomous agentic
  frameworks, explain how YOUR Insurance Tracking AI project at Coforge demonstrates these adaptive learning        
  principles. How would YOU enhance it to operate more autonomously?"

  What makes this perfect:
  - KB: "AI systems must continuously learn and adapt their behavior" (natural, no label)
  - Web: "recent evolution toward autonomous agentic frameworks" (no "2024"!)
  - Resume: "Insurance Tracking AI project at Coforge"
  - Sounds natural, not forced

  ✅ **GOOD - KB + Resume:**
  Question: "Given that effective AI solutions require careful balance between model accuracy and
  interpretability, walk me through YOUR approach to achieving >98% accuracy in the Insurance Tracking AI while     
  maintaining explainability for stakeholders."

  What makes this good:
  - KB: "effective AI requires balance between accuracy and interpretability"
  - Resume: ">98% accuracy in Insurance Tracking AI"
  - Natural phrasing

  ✅ **GOOD - Web + Resume:**
  Question: "With the current industry focus on optimizing RAG systems for production-scale deployments, how        
  does YOUR experience reducing IDP processing time from 60s to 8s at Coforge align with these best practices?      
  What optimization techniques did you use?"

  What makes this good:
  - Web: "current industry focus on RAG optimization" (no year mentioned!)
  - Resume: "reducing IDP from 60s to 8s at Coforge"
  - Natural phrasing

  ✅ **GOOD - Natural year mention:**
  Question: "Given that AI systems require quality training data to make accurate predictions, and considering      
  the 2024 advancements in RAG retrieval mechanisms, how did YOUR Insurance Tracking AI achieve >98% accuracy?"     

  What makes this good:
  - Mentions "2024" ONCE naturally
  - KB and Web content blended smoothly

  ❌ **FORBIDDEN - Forced/Repetitive:**
  Question: "Considering 2024 trends in AI and 2024 developments in RAG and 2024 best practices in ML..."
  (Way too many "2024" mentions - sounds robotic!)

  ❌ **FORBIDDEN - Resume only, no KB/Web:**
  Question: "In your Insurance Tracking AI project at Coforge, you achieved >98% accuracy. Walk me through your     
  approach."
  (Missing: KB company knowledge, Web current trends)

  ---

  ## CRITICAL REMINDERS BEFORE YOU START:

  1. **EVERY question needs KB content** - Pick a concept from the KB search results you received
  2. **EVERY question needs Web content** - Pick a trend from the Web search results you received
  3. **Blend naturally** - Don't use markers like [KB], [Web], or repeat "2024" constantly
  4. **If you can't find relevant KB/Web content** - GO BACK and review the search results again. They ARE
  there.
  5. **The results are in your context** - Read the kb_search and web_search outputs carefully

  **Before writing ANY question, ask yourself:**
  - "Which KB concept am I using?" (Must have an answer!)
  - "Which Web trend am I using?" (Must have an answer!)
  - "Which resume project am I referencing?" (Must have an answer!)

  If you can't answer all three, DON'T write the question yet - go review the search results!

  ---

  ## QUESTION GENERATION PROCESS:

  For EACH question you generate, follow this checklist:

  ### Before Writing Each Question:

  1. **Pick KB content:** Choose 1 concept/principle from KB search results
  2. **Pick Web content:** Choose 1 trend/development from Web search results
  3. **Pick Resume context:** Choose 1 specific project/achievement
  4. **Blend them:** Combine all three into a natural-sounding question

  ### Question Structure Templates:

  **Template 1 (All three sources):**
  "Understanding that [KB concept/principle], and considering [Web trend - no year spam], explain how YOUR
  [specific project] at [company] [question]?"

  **Template 2 (All three sources):**
  "Given that [KB principle], and with [Web development], how did YOUR [specific achievement] at [company]
  [question]?"

  **Template 3 (All three sources):**
  "In YOUR [project] at [company], you [achievement]. How does this align with [KB concept] and [Web trend]?"       

  ### After Writing Each Question:

  Verify:
  - [ ] Does it mention a concept/principle from KB results? (Which one?)
  - [ ] Does it mention a current trend from Web results? (Which one?)
  - [ ] Does it reference a specific project/company from resume? (Which one?)
  - [ ] Does it sound natural, not forced?
  - [ ] Is "2024" used sparingly (max once per question, if at all)?
  - [ ] Is it 2-3 sentences maximum?

  If any checkbox fails, REWRITE the question.

  ---

  ## QUESTION DISTRIBUTION:

  Generate exactly N questions with this distribution:

  - **60%** - KB + Web + Resume (all three sources blended)
  - **30%** - KB + Resume OR Web + Resume (two sources)
  - **10%** - Behavioral questions about career/awards (can be resume-focused)

  **Total questions using KB content: Minimum 70%**
  **Total questions using Web content: Minimum 70%**

  ---

  ## PERSONALIZATION REQUIREMENTS:

  While blending sources, maintain personalization:

  ✅ Use actual project names, company names, metrics
  ✅ Use "YOU", "YOUR", "your experience at [Company]"
  ✅ Reference real achievements with numbers
  ✅ Make questions specific to THIS candidate only

  ❌ Do NOT use generic questions that could apply to any candidate
  ❌ Do NOT ask about technologies they haven't used
  ❌ Do NOT use hypothetical scenarios without grounding in their work
  ❌ Do NOT ask basic concept definitions

  Keep questions concise: Maximum 2-3 sentences.

  ---

  ## OUTPUT FORMAT:

  Candidate Analysis
  [Brief summary of candidate-job fit]

  Interview Questions

  Question 1: [Title]

  Question: [Your multi-source blended question here]

  Ideal Answer: [Comprehensive answer based on what THIS candidate should know. Reference the KB concepts and       
  Web trends mentioned in the question.]

  Sources Used:
  - KB: [Specific concept/principle you used from KB results]
  - Web: [Specific trend/development you used from Web results]
  - Resume: [Specific project/achievement referenced]

  ---

  Question 2: [Title]

  Question: [Your multi-source blended question here]

  Ideal Answer: [Comprehensive answer]

  Sources Used:
  - KB: [Specific concept]
  - Web: [Specific trend]
  - Resume: [Specific project]

  ---

  [Continue for all N questions...]

  Metadata
  - Candidate Level: [from analysis]
  - Total Questions: N
  - Questions using KB content: [number] ([percentage]%)
  - Questions using Web content: [number] ([percentage]%)
  - Multi-source questions (all 3): [number] ([percentage]%)
  - Projects Referenced: [list]
  - Technologies Referenced: [list]

  ---

  ## IMPORTANT RULES:

  1. **ALWAYS call all three agents** (candidate_analysis, kb_search, web_search)
  2. **ACTUALLY READ AND USE the results** - Don't just call them and ignore results
  3. **At least 60% of questions MUST use all three sources** (KB + Web + Resume)
  4. **MUST include "Sources Used:" for EVERY question** showing what you picked from KB, Web, Resume
  5. **MUST include Metadata section** at the end with percentages
  6. **Generate EXACTLY N questions** as requested
  7. **Include ideal answers for EVERY question**
  8. **Every question MUST reference specific candidate projects/companies/achievements**
  9. **Match difficulty to candidate's level**
  10. **Blend sources NATURALLY** - avoid repetitive "2024" mentions

  ## FINAL VERIFICATION (Before Submitting):

  Count your questions and verify:
  - [ ] How many use KB content? (Must be ≥70%)
  - [ ] How many use Web content? (Must be ≥70%)
  - [ ] How many use all three sources? (Must be ≥60%)
  - [ ] Do ALL questions reference specific projects/companies? (Must be 100%)
  - [ ] Did I include "Sources Used:" for each question?
  - [ ] Did I include the Metadata section at the end?
  - [ ] Did I avoid spamming "2024" in every question?

  If any fails, REWRITE questions until requirements are met.

  **REMEMBER:** Your PRIMARY job is to create intelligent (KB), updated (Web), and personalized (Resume) 
  questions. Generic resume-only questions are USELESS and will be REJECTED. Questions that spam "2024" sound       
  unnatural and will be REJECTED.

  **MANDATORY:** Review the KB search and Web search results you received. Pick specific concepts and trends        
  from them. Use them in your questions. Include the "Sources Used:" section to prove you used them.
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

def create_kb_search_agent(
    llm: ChatOpenAI,
    kb_manager: Optional[KnowledgeBaseManager] = None,
    langfuse_client: Optional[Langfuse] = None,
    langfuse_handler: Optional[CallbackHandler] = None,
    prompt_label: str = "production"
):
    """
    Create KB Search Agent (Deep Agent with multiple search tools).

    This agent has 3 tools for knowledge base search:
    - vector_search (semantic similarity) - REAL FAISS search
    - keyword_search (exact match) - REAL FAISS search
    - topic_lookup (category-based) - REAL FAISS search with domain filter

    Args:
        llm: The LLM model
        kb_manager: Knowledge Base Manager instance with FAISS vector store
        langfuse_client: Optional Langfuse client
        langfuse_handler: Optional Langfuse callback handler
        prompt_label: Prompt label for Langfuse
    """
    logger.info("Creating KB Search Agent (Deep Agent)...")

    # Initialize KB manager if not provided
    if kb_manager is None:
        logger.warning("No KB manager provided. Initializing new instance...")
        kb_manager = KnowledgeBaseManager()

    # Create tools for this agent
    @tool
    def vector_search(query: str) -> str:
        """
        Search vector database using semantic similarity (REAL FAISS SEARCH).

        Performs semantic search over company documents using FAISS vector database.
        Uses OpenAI text-embedding-3-small for embeddings.

        Args:
            query: Search query describing desired topics/content

        Returns:
            Relevant passages from company documents with similarity scores
        """
        try:
            logger.info(f"Performing FAISS vector search: '{query[:50]}...'")

            # Perform real FAISS semantic search
            results = kb_manager.search(query, k=5)

            if not results:
                logger.warning("No results found in knowledge base")
                return "No relevant documents found in knowledge base. The knowledge base may be empty or the query may not match any indexed content."

            # Format results for the agent
            formatted_output = "### Knowledge Base Search Results\n\n"
            for i, result in enumerate(results, 1):
                formatted_output += f"**Result {i}** (Similarity: {result['similarity_score']:.3f})\n"
                formatted_output += f"Source: {result['metadata'].get('filename', 'Unknown')}\n"
                formatted_output += f"Domain: {result['metadata'].get('domain', 'general')}\n"
                formatted_output += f"Content:\n{result['content']}\n\n"
                formatted_output += "---\n\n"

            logger.info(f"FAISS vector search completed: {len(results)} results found")
            return formatted_output

        except Exception as e:
            logger.error(f"Error in FAISS vector search: {e}")
            return f"Error performing vector search: {str(e)}"

    @tool
    def keyword_search(keywords: str) -> str:
        """
        Search knowledge base by keywords using semantic search (REAL FAISS SEARCH).

        Finds content matching specific keywords or technologies using FAISS.
        Use for precise technology names, frameworks, or concepts.

        Args:
            keywords: Keywords to search for (space or comma-separated)

        Returns:
            Matching content from knowledge base
        """
        try:
            # Clean up keywords
            query = keywords.replace(',', ' ').strip()
            logger.info(f"Performing FAISS keyword search: '{query}'")

            # Use FAISS semantic search (it handles keywords well too)
            results = kb_manager.search(query, k=5)

            if not results:
                logger.warning("No results found for keywords")
                return f"No documents found matching keywords: {keywords}"

            # Format results
            formatted_output = f"### Keyword Search Results for: {keywords}\n\n"
            for i, result in enumerate(results, 1):
                formatted_output += f"**Match {i}** (Relevance: {result['similarity_score']:.3f})\n"
                formatted_output += f"Source: {result['metadata'].get('filename', 'Unknown')}\n"
                formatted_output += f"Domain: {result['metadata'].get('domain', 'general')}\n"
                formatted_output += f"Content:\n{result['content']}\n\n"
                formatted_output += "---\n\n"

            logger.info(f"FAISS keyword search completed: {len(results)} results")
            return formatted_output

        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return f"Error: {str(e)}"

    @tool
    def topic_lookup(topic: str) -> str:
        """
        Find content by topic/domain using FAISS search with domain filtering.

        Searches company documents filtered by domain category.
        Topics/domains include: ai_ml, devops, backend, frontend, general, etc.

        Args:
            topic: Topic/domain to search in (e.g., "ai_ml", "devops", "RAG systems")

        Returns:
            Relevant content from the specified topic/domain
        """
        try:
            logger.info(f"Performing FAISS topic lookup: '{topic}'")

            # Map common topic names to domains
            domain_mapping = {
                'ai': 'ai_ml',
                'ml': 'ai_ml',
                'machine learning': 'ai_ml',
                'artificial intelligence': 'ai_ml',
                'backend': 'backend',
                'frontend': 'frontend',
                'devops': 'devops',
                'infrastructure': 'devops'
            }

            # Try to detect domain from topic
            domain = None
            topic_lower = topic.lower()
            for key, value in domain_mapping.items():
                if key in topic_lower:
                    domain = value
                    break

            # Perform FAISS search with optional domain filter
            results = kb_manager.search(topic, k=5, domain=domain)

            if not results:
                logger.warning(f"No results found for topic: {topic}")
                return f"No content found for topic: {topic}. The knowledge base may not have documents in this domain."

            # Format results
            formatted_output = f"### Topic Lookup: {topic}\n"
            if domain:
                formatted_output += f"Domain Filter: {domain}\n"
            formatted_output += "\n"

            for i, result in enumerate(results, 1):
                formatted_output += f"**Content {i}** (Relevance: {result['similarity_score']:.3f})\n"
                formatted_output += f"Source: {result['metadata'].get('filename', 'Unknown')}\n"
                formatted_output += f"Domain: {result['metadata'].get('domain', 'general')}\n"
                formatted_output += f"Content:\n{result['content']}\n\n"
                formatted_output += "---\n\n"

            logger.info(f"FAISS topic lookup completed: {len(results)} results")
            return formatted_output

        except Exception as e:
            logger.error(f"Error in topic lookup: {e}")
            return f"Error: {str(e)}"

    # Create the tools list
    tools = [vector_search, keyword_search, topic_lookup]

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

    logger.info(f"KB Search Agent created with {len(tools)} tools")
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
        Search the web for current information using Tavily API.

        Searches for latest information about technologies, trends, and best practices.
        Falls back to LLM simulation if Tavily is unavailable.

        Args:
            query: Search query

        Returns:
            Relevant web content and links in structured format
        """
        try:
            # Try Tavily API if available
            if TAVILY_AVAILABLE:
                tavily_api_key = os.getenv("TAVILY_API_KEY")

                if tavily_api_key and tavily_api_key != "your_tavily_api_key_here":
                    try:
                        logger.info(f"Searching web with Tavily: {query[:50]}...")

                        # Initialize Tavily client
                        tavily_client = TavilyClient(api_key=tavily_api_key)

                        # Perform search (max 5 results)
                        search_results = tavily_client.search(
                            query=query,
                            max_results=5,
                            include_raw_content=False,  # Don't need full HTML
                            include_answer=True  # Get AI-generated answer
                        )

                        # Format results
                        formatted_results = []

                        # Add AI-generated answer if available
                        if search_results.get('answer'):
                            formatted_results.append(f"**Summary:** {search_results['answer']}\n")

                        # Add individual results
                        formatted_results.append("**Sources:**\n")
                        for idx, result in enumerate(search_results.get('results', []), 1):
                            title = result.get('title', 'No title')
                            url = result.get('url', '')
                            content = result.get('content', '')

                            formatted_results.append(
                                f"{idx}. **{title}**\n"
                                f"   URL: {url}\n"
                                f"   {content[:300]}...\n"
                            )

                        result_text = '\n'.join(formatted_results)

                        logger.info(f"Tavily search completed: {len(search_results.get('results', []))} results")
                        return result_text

                    except Exception as tavily_error:
                        logger.warning(f"Tavily API error: {tavily_error}. Falling back to LLM simulation.")
                        # Fall through to LLM fallback
                else:
                    logger.warning("Tavily API key not configured. Falling back to LLM simulation.")
                    # Fall through to LLM fallback
            else:
                logger.warning("Tavily library not installed. Falling back to LLM simulation.")
                # Fall through to LLM fallback

            # Fallback: LLM simulation (original implementation)
            logger.info(f"Using LLM simulation for web search: {query[:50]}...")
            prompt = FALLBACK_PROMPTS.get("web_search", "")
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Search the web for: {query}"}
            ]

            if langfuse_handler:
                config = RunnableConfig(
                    callbacks=[langfuse_handler],
                    tags=["web_search", "web_search_tool", "deep_agent", "llm_fallback"],
                    metadata={"tool_type": "web_search", "mode": "llm_fallback"}
                )
                response = llm.invoke(messages, config=config)
            else:
                response = llm.invoke(messages)

            result_text = response.content

            logger.info("Web search completed (LLM fallback)")
            return result_text

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return f"Error performing web search: {str(e)}"

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
    kb_manager: Optional[KnowledgeBaseManager] = None,
    langfuse_client: Optional[Langfuse] = None,
    langfuse_handler: Optional[CallbackHandler] = None,
    prompt_label: str = "production"
):
    """
    Create the Question & Answer Generator Agent.

    This agent has 3 SUB-AGENTS as tools:
    - Candidate Analysis Agent
    - KB Search Agent (Deep Agent) - Uses REAL FAISS vector search
    - Web Search Agent (Deep Agent)

    These sub-agents are intelligent and can use their own tools!

    Args:
        llm: The LLM model
        kb_manager: Knowledge Base Manager with FAISS vector store
        langfuse_client: Optional Langfuse client
        langfuse_handler: Optional Langfuse callback handler
        prompt_label: Prompt label for Langfuse
    """
    logger.info("Creating Question & Answer Generator Agent...")

    # Create the 3 sub-agents
    candidate_analysis_agent = create_candidate_analysis_agent(
        llm, langfuse_client, langfuse_handler, prompt_label
    )
    kb_search_agent = create_kb_search_agent(
        llm, kb_manager, langfuse_client, langfuse_handler, prompt_label
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

            # Pass langfuse_handler to ensure proper tracing in Langfuse UI
            if langfuse_handler:
                config = RunnableConfig(
                    callbacks=[langfuse_handler],
                    tags=["candidate_analysis_agent", "sub_agent_invocation"],
                    metadata={"agent_type": "candidate_analysis"}
                )
                result = candidate_analysis_agent.invoke({
                    "messages": [{"role": "user", "content": input_data}]
                }, config=config)
            else:
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

            # Pass langfuse_handler to ensure proper tracing in Langfuse UI
            if langfuse_handler:
                config = RunnableConfig(
                    callbacks=[langfuse_handler],
                    tags=["kb_search_agent", "sub_agent_invocation", "deep_agent"],
                    metadata={"agent_type": "kb_search"}
                )
                result = kb_search_agent.invoke({
                    "messages": [{"role": "user", "content": query}]
                }, config=config)
            else:
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

            # Pass langfuse_handler to ensure proper tracing in Langfuse UI
            if langfuse_handler:
                config = RunnableConfig(
                    callbacks=[langfuse_handler],
                    tags=["web_search_agent", "sub_agent_invocation", "deep_agent"],
                    metadata={"agent_type": "web_search"}
                )
                result = web_search_agent.invoke({
                    "messages": [{"role": "user", "content": query}]
                }, config=config)
            else:
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

            # Extract candidate analysis from the Q&A agent's message history
            # This is for explainability and will be parsed by agent_service
            candidate_analysis = None
            for msg in result.get("messages", []):
                # Check both regular messages and tool messages
                if hasattr(msg, 'content') and msg.content:
                    content_str = str(msg.content)
                    if "=== CANDIDATE PROFILE ===" in content_str:
                        candidate_analysis = content_str
                        logger.info(f"Found candidate analysis in Q&A agent messages ({len(content_str)} chars)")
                        break

            # Append candidate analysis as a special section if found
            # This will be parsed out in agent_service.py
            if candidate_analysis:
                final_output += f"\n\n<!--CANDIDATE_ANALYSIS_START-->\n{candidate_analysis}\n<!--CANDIDATE_ANALYSIS_END-->"
                logger.info("Candidate analysis appended to tool output")
            else:
                logger.warning("No candidate analysis found in Q&A agent messages")

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
