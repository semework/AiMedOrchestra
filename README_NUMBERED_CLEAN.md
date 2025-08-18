# AiMedOrchestra — Overview + Setup & Run Guide

> A virtual AI hospital you can run on a laptop — multiple specialist agents coordinated by a conversational orchestrator.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Mini Pipeline (3 Steps)](#2-mini-pipeline-3-steps)
3. [Installation & Quick Start](#3-installation--quick-start)
   - [3.1 Clone & Create a Virtual Env](#31-clone--create-a-virtual-env)
   - [3.2 Install Dependencies](#32-install-dependencies)
   - [3.3 Run the Conversational Notebook](#33-run-the-conversational-notebook)
   - [3.4 Quick Conversational REPL](#34-quick-conversational-repl)
   - [3.5 Run Per‑Agent Demo Apps](#35-run-peragent-demo-apps)
   - [3.6 Docker (Optional)](#36-docker-optional)
   - [3.7 Kubernetes (Optional)](#37-kubernetes-optional)
   - [3.8 Data & Samples](#38-data--samples)
   - [3.9 Troubleshooting](#39-troubleshooting)
4. [Summary](#4-summary)
5. [References](#5-references)
6. [License](#6-license)

---

## 1) Overview

Swap the trivia question for a patient’s **symptoms**, **labs**, **imaging**, or **genomics**.  
**AiMedOrchestra** routes your natural sentence to the right specialist agent(s) and stitches results into a single response.

- The **orchestrator** reads your request (“create two synthetic patients”, “trials near Boston”) and picks the right agent(s).
- Each **agent** is a focused specialist (diagnostics, imaging, genomics, trials, diet/treatment, literature, mental‑health support, ethics).
- Outputs from one agent can feed the next, giving you an end‑to‑end answer in plain language.

---

## 2) Mini Pipeline (3 agents)

| Agent | Agent / Model | Role | Input | Output |
| --- | --- | --- | --- | --- |
| 1 | Data Synthesis | Create synthetic patient records | N = 2 | Two JSON rows (synthetic patients) |
| 2 | Diagnostics | Suggest likely diagnosis from symptoms | 55F, chest pain + cough | Differential (e.g., bronchitis, pneumonia) |
| 3 | Clinical Trial Matching | Find recruiting trials for a given profile | 65M, lung cancer, Boston | Trial shortlist (e.g., NCT012345, NCT067890) |


> **How to use it:** Type natural instructions. The orchestrator handles routing (e.g., “**create 2 synthetic patients**”, “**diagnose a 55 yo female with chest pain**”, “**trials for lung cancer near Boston**”).

---

## 3) Installation & Quick Start

### 3.1 Clone & Create a Virtual Env
```bash
git clone https://github.com/semework/AiMedOrchestra.git
cd AiMedOrchestra

# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 3.2 Install Dependencies
Use either `requirements.txt` or editable install via `pyproject.toml`:

**Option A — requirements.txt**
```bash
pip install -r requirements.txt
```

**Option B — editable install (preferred for dev)**
```bash
pip install -e .
```
> Editable install lets you `import aimedorchestra...` without fiddling with `PYTHONPATH`.

### 3.3 Run the Conversational Notebook
```bash
jupyter lab  # or: jupyter notebook
# Open: aimedorchestra_demo_notebook_chat.ipynb
# Run the "Chat UI for AiMedOrchestra" cell, then try:
#   - "create two synthetic patients"
#   - "diagnose a 55 yo female with chest pain and cough"
#   - "find clinical trials for lung cancer near Boston"
#   - 'run full pipeline on {"age":60, "sex":"male"}'
```

### 3.4 Quick Conversational REPL
**If installed with `pip install -e .`:**
```bash
python - << 'PY'
from aimedorchestra.orchestrator.agent import aimedorchestraAgent

orch = aimedorchestraAgent()
print(orch.route("create two synthetic patients"))
print(orch.route("diagnose a 55 yo female with chest pain and cough"))
print(orch.route("find clinical trials for lung cancer near Boston"))
print(orch.route('run full pipeline on {"age":60,"sex":"male","conditions":["diabetes"]}'))
PY
```

**Alternative (use top‑level file from repo root):**
```bash
PYTHONPATH=. python - << 'PY'
from aimedorchestraagent import aimedorchestraAgent
print(aimedorchestraAgent().route("create 3 synthetic patients"))
PY
```

### 3.5 Run Per‑Agent Demo Apps

**Streamlit**
> If the app can’t import the package, prefix with `PYTHONPATH=.`
```bash
PYTHONPATH=. streamlit run apps/orchestrator_app.py
# or per-agent:
# PYTHONPATH=. streamlit run apps/data_synthesis_app.py
# PYTHONPATH=. streamlit run apps/diagnostics_app.py
# ...
```

**Pure Python / Dash**
```bash
python apps/orchestrator_app.py
# or per-agent:
# python apps/data_synthesis_app.py
# python apps/diagnostics_app.py
# ...
```
> Streamlit typically uses **8501**; Dash commonly uses **8050**.

### 3.6 Docker (Optional)
```bash
docker build -t aimed:latest .
# If Streamlit (8501):
docker run --rm -p 8501:8501 aimed
# If Dash/Flask (8050):
# docker run --rm -p 8050:8050 aimed
```

### 3.7 Kubernetes (Optional)
```bash
# Local K8s
kind create cluster --name aimed

# Deploy
kubectl apply -f k8s_deployment.yaml

# Check rollout
kubectl get pods
kubectl logs deploy/aimedorchestra -f
# Optional port-forward:
kubectl port-forward deploy/aimedorchestra 8501:8501
```

### 3.8 Data & Samples
- Sample image: `data/sample_image.jpg`  
- Trials JSON: `data/trials.json`  
- Literature snippets: `data/literature/*.txt`

### 3.9 Troubleshooting
- If the package can’t be found, run with `PYTHONPATH=.` from the repo root, e.g.:
  ```bash
  PYTHONPATH=. python apps/orchestrator_app.py
  ```
- For Streamlit imports, use:
  ```bash
  PYTHONPATH=. streamlit run apps/orchestrator_app.py
  ```
- If an agent import fails, the router returns `(agent_unavailable)`. Ensure `aimedorchestra/agents/<agent>/agent.py` exists and its class ends with `Agent`.

---

## 4) Summary
AiMedOrchestra is a team of specialized AI agents coordinated by an orchestrator so you can **chat** your way through complex clinical tasks and get a single, coherent answer.

---

## 5) References
(See project README or paper references as applicable.)

---

## 6) License
Released for non‑commercial & research use. See the repository’s LICENSE file for details.
