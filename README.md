# Kubernetes Incident Support Assistant

A buildathon project demonstrating a sequential multi-agent workflow using Microsoft's AutoGen AgentChat API.

## Scenario

A DevOps engineer reports a Kubernetes incident such as:

> Our Kubernetes application is showing CrashLoopBackOff. What could be the reason and how should I troubleshoot it?

Three AutoGen `AssistantAgent`s process the request sequentially.

## Architecture

```text
DevOps Engineer
      |
      v
+----------------------+
| 1. DevOps Assistant  |
| LLM knowledge only   |
| No tools             |
+----------+-----------+
           |
           v
+--------------------------+
| 2. Web Search Assistant  |
| SerpApi tool             |
| Current web information  |
+------------+-------------+
             |
             v
+--------------------------+
| 3. Entry Agent          |
| save_incident() tool    |
| Query + Answer 1 + 2    |
+------------+-------------+
             |
             v
       Streamlit UI
```

## Key learning

The project deliberately demonstrates:

- LLM knowledge vs live web-grounded information
- AutoGen `AssistantAgent`
- `RoundRobinGroupChat`
- Sequential agent execution
- Tool assignment per agent
- Termination conditions
- Streamlit integration
- Persisting an interaction to a `.txt` file

## Agent responsibilities

### 1. DevOps Assistant

Uses only the LLM's existing knowledge.

No tools are provided.

### 2. Web Search Assistant

Has the `web_search()` Python function.

Uses SerpApi to search current information and returns a web-grounded response.

### 3. Entry Agent

Has the `save_incident()` Python function.

Saves:

- Original incident
- Answer 1
- Answer 2

into a timestamped text file.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=...
OPENAI_BASE_URL=...
SERPAPI_API_KEY=...
```

Do not commit `.env`.

## Run

```bash
streamlit run app.py
```

## Example

Input:

```text
Our Kubernetes application is showing CrashLoopBackOff.
What could be the reason and how should I troubleshoot it?
```

The first agent answers using model knowledge.

The second agent searches current Kubernetes information.

The third agent saves both responses.

Files are written under:

```text
outputs/
```

## Important

This is a learning/buildathon project. It does not connect to a live Kubernetes cluster. The web-search agent provides current web information, while the first agent provides an LLM-knowledge perspective.

## Flow

<img width="1225" height="1284" alt="k8s autogeb agent" src="https://github.com/user-attachments/assets/84d4c01e-e81f-40b7-99ac-99b9b9319967" />

