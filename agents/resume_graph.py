from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# ---------- Load Environment Variables ----------
load_dotenv()

# ---------- Define Graph State Schema ----------
class ResumeState(BaseModel):
    resume: str
    job_desc: str
    analysis: str | None = None
    letter: str | None = None

# ---------- LLM Setup ----------
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # lightweight Gemini model
    temperature=0.7,
    top_p=0.85,
    google_api_key=os.getenv("GOOGLE_API_KEY")  # ✅ safer: load from .env
)

# ---------- 1️⃣ Resume Analysis Node ----------
def analyze_resume_node(state: ResumeState):
    resume = state.resume
    job_desc = state.job_desc

    prompt = ChatPromptTemplate.from_template("""
    You are an expert technical HR analyst.
    Evaluate the following RESUME against the JOB DESCRIPTION.

    === RESUME ===
    {resume}

    === JOB DESCRIPTION ===
    {job_desc}

    Return the output in structured Markdown format:

    Candidate Strengths
    (Summarize in 3–5 bullet points)

    Areas for Improvement
    (Give precise actionable suggestions)

    Role Fit Recommendation
    (Suggest 2–3 best-matched job titles)
    """)

    chain = prompt | gemini_llm
    analysis_text = chain.invoke({"resume": resume, "job_desc": job_desc}).content

    state.analysis = analysis_text
    return state


# ---------- 2️⃣ Cover Letter Node ----------
def cover_letter_node(state: ResumeState):
    resume = state.resume
    job_desc = state.job_desc

    prompt = ChatPromptTemplate.from_template("""
    You are a professional career writer.
    Using the resume and job description below, generate a short (150–200 word)
    cover letter highlighting the top 2–3 relevant skills and achievements.

    === RESUME ===
    {resume}

    === JOB DESCRIPTION ===
    {job_desc}

    Format:
    Dear Hiring Manager,
    [Body paragraphs]
    Sincerely,
    [Candidate Name]
    """)

    chain = prompt | gemini_llm
    letter_text = chain.invoke({"resume": resume, "job_desc": job_desc}).content

    state.letter = letter_text
    return state


# ---------- 3️⃣ Build LangGraph ----------
def create_resume_analysis_graph():
    # ✅ Include the required `state_schema`
    graph = StateGraph(state_schema=ResumeState)

    # Add nodes
    graph.add_node("resume_analysis", analyze_resume_node)
    graph.add_node("cover_letter", cover_letter_node)

    # Define flow
    graph.set_entry_point("resume_analysis")
    graph.add_edge("resume_analysis", "cover_letter")
    graph.add_edge("cover_letter", END)

    # Compile the graph
    return graph.compile()
