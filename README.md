# Meet **AI Med Orchestra** : A Virtual AI Hospital for Smarter Healthcare
*A virtual hospital you can run on a laptop – eleven specialised agents, one cooperative care team.*
## CTGAN / FLAN-T5 / ResNet / DNABERT / GPT-GNN in one AI hospital

## What is AI Med Orchestra and Why Does It Matter?
Imagine if every department in a modern hospital had an AI colleague — one that never sleeps, never forgets, and constantly studies the newest medical breakthroughs. That’s the vision behind **AiMedOrchestra** (also known as OmniHospital AI): an open-source, multi-agent platform that bundles **eleven specialized AIs** — each a board-certified genius in its own domain — into a single, cooperative care team that can run on your laptop.

Think of it as a virtual hospital made up of tireless, super-smart assistants, each expert in a different medical field, working together seamlessly to give the best possible care.  

Why is this important? Because healthcare is both **extremely complex** and **fast-changing**. Doctors and nurses must juggle enormous volumes of information — patient histories, lab results, medical images, genetics, research papers, and treatment guidelines. No single human can keep up with it all, but AI can help. AiMedOrchestra’s coordinated team of AI agents empowers clinicians to manage complexity, stay current with medical advances, and deliver better patient outcomes.

First, let's be on the same page with the main concepts and how we got here ...

---

## What Are LLMs and AI Agents, Simply Put?

**Large Language Models (LLMs)**  
These are computer programs trained on enormous amounts of text, like books, websites, and medical journals. They learn patterns in language and can answer questions, write explanations, or generate text that sounds human. Think of them as very advanced virtual assistants that understand and generate language. Agents, or a super photgraphic (AKA Eidetic memory) friend :) Reasoning and context are a whole other discussion!

**AI Agents**  
An “agent” is like a mini-expert robot inside your computer. It listens to questions or data, thinks about it using specialized skills or knowledge, and then provides answers or takes action. Unlike one big AI trying to do everything, agents are specialized — one might be great at reading X-rays, another at understanding genetics, another at mental health conversations.

**Multi-Agent Systems**  
Imagine a team of experts each doing their part and sharing notes. The final decision or advice comes from combining all their expertise. That’s what AiMedOrchestra does by connecting 11 different agents into one “care team.”

---

## Simple Example: How AI Agents and LLMs Work Together to Answer a Question

Suppose you ask:  
**“What is the capital of California?”**

Here’s how a simple multi-agent system might answer:

| Step | Agent / Model         | Role                              | Input                                 | Output                                  |
|------|-----------------------|-----------------------------------|----------------------------------------|------------------------------------------|
| 1    | Language Model Agent  | Understand the question           | “What is the capital of California?”   | Recognizes it’s a geography question     |
| 2    | Search Agent          | Look up facts                     | Query: “Capital of California”         | Retrieves “Sacramento”                   |
| 3    | LLM Agent             | Create an easy-to-understand reply| Fact: “Sacramento”                      | “The capital of California is Sacramento.”|

The agents work together by passing information: The first agent understands the question’s meaning, the second finds the correct fact, and the third turns it into a natural sentence, as visualized in the diagram below.

---

<img src="https://github.com/semework/AiMedOrchestra/blob/main/assets/LLM_workflow.png" style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 95%;"/> 

This simple flow shows how specialized AI pieces (“agents”) handle different parts of the task, working together smoothly.

---

## How Does This Scale to Healthcare?
Replace “capital of California” with a patient’s symptoms, lab tests, imaging scans, or genetic data. Replace “search agent” with AI trained on medical images or clinical trial databases. Now you have a multi-agent AI hospital like AiMedOrchestra that can:

- Suggest diagnoses from symptoms and imaging  
- Plan personalized diets and treatments  
- Match patients to clinical trials  
- Monitor fairness and ethics in AI decisions  

And much more — all on your laptop!

---

<p align="center"> <img src="https://github.com/semework/AiMedOrchestra/blob/main/assets/demo.gif" width="90%" alt="OmniHospital demo"> </p> 

> **Practice notebook**: see **`aimedorchestra_demo_notebook.ipynb`** for a guided tour with a router diagram, inline chat, and smoke‑tests.

---

### Table of Contents
1. [Vision & Motivation](#1)  
2. [Meet the Agents](#2)  
3. [Agent Details](#3)  
4. [System Architecture](#4)  
5. [Installation & Quick Start](#5)  
6. [Summary](#5) 
7. [References](#refs)

---


## 1 Vision & Motivation
Healthcare knowledge doubles every **73 days**. Clinicians juggle guidelines, imaging, multi-omics, trial eligibility and evolving goals — often in legacy EMRs. Point AI tools help, but live in silos.

**AIMedOrchestra** unifies eleven narrow yet excellent agents into a digital care team:
* **Edge-friendly** — a laptop or single-node K8s cluster is enough.

---

## 2 Meet the Agents
<img src="https://github.com/semework/AiMedOrchestra/blob/main/assets/router.png" style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 95%;"/> 
| ID   | Service              | Core Models                                   | Input → Output                                          |
|------|----------------------|-----------------------------------------------|---------------------------------------------------------|
| **A1**  | Data Synthesis         | CTGAN · VAE-Survival                        | schema → synthetic EHR rows                             |
| **A2**  | Diagnostics LLM       | FLAN-T5-XXL + rules                          | vitals + notes + findings → ranked diagnoses            |
| **A3** | Diet Planner          | GPT-2 text-gen (HF pipeline)                | patient profile → 7-day personalized meal plan          |
| **A4**  | Drug Discovery GNN    | PyG GIN · GPT-GNN                            | target protein → 10 candidate molecules                 |
| **A5** | Ethics & Bias Auditor | Fairlearn metrics · SHAP                    | decision logs → bias report                             |
| **A6**  | Genomics              | DNABERT · ESM-1b                            | VCF → ACMG classification                               |
| **A7**  | Imaging               | ResNet-50 · EfficientNet-b3 · Grad-CAM      | DICOM/PNG → label + heat-map                            |
| **A8**  | Literature RAG        | Sentence-BERT + T5                           | query → summary + citations                             |
| **A9**  | Mental-Health Chat    | DialoGPT-medium · RoBERTa-sent              | text turn → empathic reply + sentiment                  |
| **A10**  | Treatment RL          | PPO (stable-baselines 3)                    | state vector → optimized dose plan                      |
| **A11**  | Trial Matcher         | mini-BERT + regex                            | patient snapshot → recruiting trial IDs                 |


## 3 Agent Details

### Data Synthesizer (A1)
Generates realistic synthetic EHR rows with **CTGAN**. Trains on tabular distributions, preserves utility while removing PHI.

### Diagnostics LLM (A2)
Fuses history, imaging and genomics into an AI‑written differential; returns ranked diagnoses and next‑step recommendations.

### Diet Planner (A3)
Generates 7-day meal plans from medical conditions and preferences using GPT-2. Tailors dietary advice to support comorbidities and lifestyle constraints.

### Drug‑Discovery GNN (A4)
Graph Neural pipeline ranks molecules for a disease target and explains reasoning.

### Ethics & Bias Auditor (A5)
Runs fairness metrics on synthetic twins; logs SHAP explanations.

### Genomics Classifier (A6)
DNABERT + ESM‑1b predict ACMG pathogenicity and flag actionable mutations.

### Imaging Agent (A7)
ResNet‑50 + EfficientNet‑b3 ensemble with Grad‑CAM. Outputs labels and heat‑maps for rapid visual verification.

### Literature RAG (A8)
SBERT retrieval + T5 summarization → evidence bullets with citations.

### Mental‑Health Chat (A9)
DialoGPT dialogue plus emotion classifier; outputs empathetic text + cue.

### Treatment RL (A10)
PPO policy suggests safe No/Low/Med/High dosing plans from raw patient profiles.

### Trial Matcher (A11)
Screens ClinicalTrials.gov recruiting studies, returns NCT IDs, titles, links.

---

### 4 System Architecture — *AiMedOrchestra*

| Layer 📦 | Technology / Notes |
|----------|--------------------|
| **Orchestrator** | **LangGraph** finite-state router (Python, hot-reload friendly) |
| **Message Bus** | **Redis Streams + JSONSchema** (default) · swap-in **NATS** for multi-cluster deployments |
| **Agents** | One **Docker** image per agent <br>— semantic-versioned (`v1.x.y`) and health-checked at `/healthz` |
| **Storage** | **MinIO** (object blobs — DICOM, PDFs, models) <br>**PostgreSQL** (metadata & audit log) |
| **Deploy** | GitHub Actions → Docker Hub CI/CD <br>**Helm chart** in `/k8s/helm/` for K8s / KinD |
| **Observability** | **Prometheus** + **Grafana** dashboards (CPU, GPU, per-agent latency) |

> **Minimum dev rig:** 8 vCPU · 24 GB RAM · NVIDIA RTX 3060 (12 GB)<br>
> *Production scales horizontally via Kubernetes replicas.*

---
## 5. Summary
- **AiMedOrchestra** is like a team of AI doctors and specialists working together inside your computer.  
- It uses many AI “agents,” each an expert in a healthcare domain, collaborating seamlessly.  
- This approach helps doctors manage complex information, improve care, and keep up with rapid medical advances.  
- By running on accessible hardware, it brings cutting-edge AI hospital capabilities into any clinical or research setting.

## 6 Installation & Quick Start

## Quick start

```bash
git clone https://github.com/semework/AiMedOrchestra.git
cd aimedorchestra-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py   # ~6 GB of open-source weights
python main.py                      # CLI walk-through
 
docker compose up --build
open http://localhost:8501      # Streamlit dashboard
 
kind create cluster --name aimed
kubectl apply -f kubernetes/base/
kubectl apply -f kubernetes/agents/
 ```
License
Released under the Business Source License 1.1 – free for non-commercial & research use; commercial use requires a license after the Change Date.

---
## 7 References
 
| Ref | Citation | Link |
|-----|----------|------|
| 1 | Johnson, A. E. W. *et al.* “MIMIC-IV.” *PhysioNet* (dataset), 2023. | <https://physionet.org/content/mimiciv/2.2> |
| 2 | Rajpurkar, P. *et al.* “CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.” *NeurIPS ML4H Workshop*, 2017. | <https://arxiv.org/abs/1711.05225> |
| 3 | Rives, A. *et al.* “Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences.” *PNAS* 118 (15), 2021. | <https://www.pnas.org/doi/10.1073/pnas.2016239118> |
| 4 | Huang, K.; Chandak, P.; Wang, Q. *et al.* “A Foundation Model for Clinician-Centered Drug Repurposing.” *Nature Medicine* 30, 3601-3613 (2024). | <https://www.nature.com/articles/s41591-024-03233-x> |
| 5 | Jin, Q.; Wang, Z.; Floudas, C. S. *et al.* “Matching Patients to Clinical Trials with Large Language Models.” *Nature Communications* 15, Article 53081 (2024). | <https://www.nature.com/articles/s41467-024-53081-z> |
| 6 | Mehrabi, N. *et al.* “A Survey on Bias and Fairness in Machine Learning.” *ACM Computing Surveys* 54 (6), 2021. | <https://arxiv.org/abs/1908.09635> |
| 7 | Densen, B. “Challenges and Opportunities Facing Medical Education.” *Transactions of the American Clinical and Climatological Association* 122, 2011. | <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3116346> |
---

If you use AiMedOrchestra in research, please cite:

Mulugeta Semework Abebe. 2025. “AiMedOrchestra: A Multi-Agent Platform for Full Med Care.” 
