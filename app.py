import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ============================================================
# Configuration
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Kubernetes Incident Support Assistant",
    page_icon="🚨",
    layout="wide",
)

st.markdown(
    """
    <style>
    .title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    .card {
        padding: 1rem 1.2rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #fafafa;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title">🚨 Kubernetes Incident Support Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AutoGen sequential multi-agent workflow: LLM knowledge → Live web search → Incident entry"
    "</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 🧠 Agent 1")
    st.caption("DevOps Assistant — LLM knowledge")

with c2:
    st.markdown("### 🌐 Agent 2")
    st.caption("Web Search Assistant — SerpApi")

with c3:
    st.markdown("### 📝 Agent 3")
    st.caption("Entry Agent — saves incident")

st.divider()


# ============================================================
# Model Client
# ============================================================

def create_model_client() -> OpenAIChatCompletionClient:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing in .env")

    if not OPENAI_MODEL:
        raise ValueError("OPENAI_MODEL is missing in .env")

    kwargs: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "api_key": OPENAI_API_KEY,
        "temperature": 0.2,
        "model_info": {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    }

    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    return OpenAIChatCompletionClient(**kwargs)


# ============================================================
# TOOL 1
# Web Search Assistant ONLY
# ============================================================

def web_search(query: str) -> str:
    """
    Search current web information using SerpApi.
    This tool is available ONLY to the Web Search Assistant.
    """

    if not SERPAPI_API_KEY:
        return (
            "WEB_SEARCH_ERROR: SERPAPI_API_KEY is missing. "
            "Web search could not be performed."
        )

    query = query.strip()

    if not query:
        return "WEB_SEARCH_ERROR: Empty search query."

    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "num": 5,
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()
        results = data.get("organic_results", [])

        if not results:
            return f"No web results found for: {query}"

        output = []

        for index, result in enumerate(results[:5], start=1):
            title = result.get("title", "Untitled")
            snippet = result.get("snippet", "")
            link = result.get("link", "")

            output.append(
                f"{index}. {title}\n"
                f"   Snippet: {snippet}\n"
                f"   URL: {link}"
            )

        return "\n\n".join(output)

    except requests.RequestException as exc:
        return f"WEB_SEARCH_ERROR: {exc}"

    except Exception as exc:
        return f"WEB_SEARCH_ERROR: {exc}"


# ============================================================
# TOOL 2
# Entry Agent ONLY
# ============================================================

def save_incident(
    query: str,
    answer_1: str,
    answer_2: str,
) -> str:
    """
    Save the incident query and both agent responses.

    This tool is available ONLY to the Entry Agent.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_path = OUTPUT_DIR / f"kubernetes_incident_{timestamp}.txt"

    content = f"""
KUBERNETES INCIDENT SUPPORT ASSISTANT
=====================================

Timestamp:
{datetime.now().isoformat(timespec="seconds")}

USER INCIDENT
-------------
{query}

ANSWER 1 - DEVOPS ASSISTANT
---------------------------
{answer_1}

ANSWER 2 - WEB SEARCH ASSISTANT
-------------------------------
{answer_2}

=====================================
END OF INCIDENT
=====================================
""".strip()

    file_path.write_text(content, encoding="utf-8")

    return f"Incident saved successfully to: {file_path}"


# ============================================================
# AutoGen Agents
# EXACTLY THREE AssistantAgents
# ============================================================

def create_agents(model_client: OpenAIChatCompletionClient):

    # --------------------------------------------------------
    # Agent 1
    # --------------------------------------------------------

    devops_assistant = AssistantAgent(
        name="devops_assistant",
        model_client=model_client,
        system_message="""
You are the DevOps Assistant.

Your responsibility is to answer the user's Kubernetes or
DevOps incident using ONLY your existing LLM knowledge.

IMPORTANT:
- You have NO tools.
- Do NOT perform web searches.
- Do NOT pretend that you searched the web.
- Do NOT use information from the Web Search Assistant.
- Provide a practical troubleshooting answer.
- Explain likely causes and troubleshooting steps.
- Clearly distinguish general Kubernetes knowledge from
  information that would require checking the live cluster.
- Do not claim that you can see the user's Kubernetes cluster.

Example:
If the user reports CrashLoopBackOff, explain common causes
such as application crashes, configuration problems,
environment variables, probes, dependencies and resource
issues, and provide kubectl commands that the engineer could
run manually.

Your response will be passed to the next agent.
""",
    )

    # --------------------------------------------------------
    # Agent 2
    # --------------------------------------------------------

    web_search_assistant = AssistantAgent(
        name="web_search_assistant",
        model_client=model_client,
        tools=[web_search],
        system_message="""
You are the Web Search Assistant.

You are the ONLY agent that has access to the web_search tool.

Your responsibility is to:
1. Understand the original Kubernetes/DevOps incident.
2. Search the current web using the web_search tool.
3. Prefer authoritative and current sources such as:
   - Kubernetes documentation
   - CNCF documentation
   - Official vendor documentation
   - Other reputable technical sources
4. Produce a web-grounded troubleshooting answer.
5. Include useful source URLs when available.

IMPORTANT:
- You MUST use the web_search tool before producing your answer.
- Do NOT claim that you searched if the tool failed.
- Do NOT use file-writing tools.
- Do NOT modify or invent the first agent's response.
- Focus on current information and documentation.

Your response will be passed to the Entry Agent.
""",
    )

    # --------------------------------------------------------
    # Agent 3
    # --------------------------------------------------------

    entry_agent = AssistantAgent(
        name="entry_agent",
        model_client=model_client,
        tools=[save_incident],
        system_message="""
You are the Entry Agent and the FINAL agent in this workflow.

You are the ONLY agent with access to the save_incident tool.

The conversation contains:
1. The original user incident.
2. Answer 1 from the DevOps Assistant.
3. Answer 2 from the Web Search Assistant.

Your responsibilities:

1. Extract the original user query.
2. Extract Answer 1.
3. Extract Answer 2.
4. Call save_incident EXACTLY ONCE.
5. Pass the original query, Answer 1 and Answer 2 to the tool.
6. After the tool succeeds, provide a concise final message.

Do NOT:
- Rewrite Answer 1.
- Rewrite Answer 2.
- Invent additional technical information.
- Perform web searches.
- Diagnose a live Kubernetes cluster.
- Use any other tool.

Your final message must contain:
- Confirmation that the incident was saved.
- Answer 1.
- Answer 2.

At the very end, write:

ENTRY_COMPLETE
""",
    )

    return devops_assistant, web_search_assistant, entry_agent


# ============================================================
# Message Helpers
# ============================================================

def content_to_text(content: Any) -> str:

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))

            elif hasattr(item, "text"):
                parts.append(str(item.text))

            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def message_to_text(message: Any) -> str:

    return content_to_text(
        getattr(message, "content", message)
    )


def extract_messages(result: Any):

    messages = getattr(result, "messages", []) or []

    extracted = []

    for message in messages:

        source = getattr(message, "source", "") or ""
        content = message_to_text(message).strip()

        if content:
            extracted.append(
                {
                    "source": source,
                    "content": content,
                }
            )

    return extracted


def latest_agent_response(messages, agent_name: str) -> str:

    for message in reversed(messages):

        if message["source"] == agent_name:

            return message["content"]

    return ""


# ============================================================
# Workflow
# ============================================================

async def run_workflow(query: str):

    model_client = create_model_client()

    devops_assistant, web_search_assistant, entry_agent = create_agents(
        model_client
    )

    # The assignment explicitly requires:
    # RoundRobinGroupChat
    #
    # Order:
    # 1. DevOps Assistant
    # 2. Web Search Assistant
    # 3. Entry Agent

    termination = (
        TextMentionTermination("ENTRY_COMPLETE")
        | MaxMessageTermination(3)
    )

    team = RoundRobinGroupChat(
        participants=[
            devops_assistant,
            web_search_assistant,
            entry_agent,
        ],
        termination_condition=termination,
    )

    try:

        result = await team.run(task=query)

        return result

    finally:

        await model_client.close()


# ============================================================
# Environment Validation
# ============================================================

def validate_environment():

    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if not OPENAI_MODEL:
        missing.append("OPENAI_MODEL")

    if not SERPAPI_API_KEY:
        missing.append("SERPAPI_API_KEY")

    return missing


# ============================================================
# Streamlit Input
# ============================================================

st.subheader("Describe the Kubernetes incident")

query = st.text_area(
    "Incident / Task",
    placeholder=(
        "Example:\n"
        "Our Kubernetes application is showing "
        "CrashLoopBackOff. What could be the reason "
        "and how should I troubleshoot it?"
    ),
    height=140,
)

run_button = st.button(
    "🚀 Analyze Incident",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Execute
# ============================================================

if run_button:

    if not query.strip():

        st.warning("Please enter a Kubernetes incident.")

        st.stop()

    missing = validate_environment()

    if missing:

        st.error(
            "Missing environment variables: "
            + ", ".join(missing)
        )

        st.stop()

    with st.spinner(
        "Running DevOps Assistant → Web Search Assistant → Entry Agent..."
    ):

        try:

            result = asyncio.run(
                run_workflow(query.strip())
            )

            messages = extract_messages(result)

            answer_1 = latest_agent_response(
                messages,
                "devops_assistant",
            )

            answer_2 = latest_agent_response(
                messages,
                "web_search_assistant",
            )

            entry_response = latest_agent_response(
                messages,
                "entry_agent",
            )

            st.session_state["result"] = {
                "query": query.strip(),
                "answer_1": answer_1,
                "answer_2": answer_2,
                "entry_response": entry_response,
            }

        except Exception as exc:

            st.error(
                f"Workflow failed: {exc}"
            )

            st.exception(exc)

            st.stop()


# ============================================================
# Results
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.subheader("🔎 Incident Analysis")

    st.markdown(
        f"""
        <div class="card">
        <strong>Incident:</strong><br>
        {result["query"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🧠 Answer 1 — DevOps Assistant"
        )

        st.markdown(
            '<div class="card">'
            + result["answer_1"]
            + "</div>",
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            "### 🌐 Answer 2 — Web Search Assistant"
        )

        st.markdown(
            '<div class="card">'
            + result["answer_2"]
            + "</div>",
            unsafe_allow_html=True,
        )

    st.success(
        "📝 Entry Agent completed the file-writing step. "
        "Check the outputs/ directory for the timestamped incident file."
    )

    with st.expander("Entry Agent status"):

        st.write(
            result["entry_response"]
            .replace("ENTRY_COMPLETE", "")
            .strip()
        )
