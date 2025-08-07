# AI Med Orchestra
*A virtual hospital you can run on a laptop – ten specialised agents, one cooperative care team.*

## CTGAN / FLAN-T5 / ResNet / DNABERT / GPT-GNN in one AI hospital

<p align="center"> <img src="https://github.com/semework/AiMedOrchestra/blob/main/assets/demo.gif" width="90%" alt="OmniHospital demo"> </p> 

> *“What if every department in a modern hospital had an AI colleague—
> one that never sleeps, never forgets, and constantly studies the
> newest medical breakthroughs?”*  
> That vision is OmniHospital AI, an open-source **multi-agent** platform
> that bundles ten specialized AIs—each a board-certified genius in
> its own domain—into a single, cooperative care team.

> **Practice notebook**: see **`aimedorchestra_demo_notebook.ipynb`** for a guided tour with a router diagram, inline chat, and smoke‑tests.

---

### Table of Contents
1. [Vision & Motivation](#1)  
2. [Meet the Agents](#2)  
3. [Agent Details](#3)  
4. [System Architecture](#4)  
5. [Installation & Quick Start](#5)  
6. [Agent APIs](#6)  
7. [Data · Security · Privacy](#7)  
8. [Evaluation & Benchmarks](#8)  
9. [Limitations & Roadmap](#9)  
10. [Contributing & Community](#10)  
11. [License](#license)  
12. [References](#refs)

---


<a id="1"></a>
## 1 Vision & Motivation
Healthcare knowledge doubles every **73 days**. Clinicians juggle guidelines, imaging, multi-omics, trial eligibility and evolving goals — often in legacy EMRs. Point AI tools help, but live in silos.

**AIMedOrchestra** unifies ten narrow yet excellent agents into a digital care team:
* **Edge-friendly** — a laptop or single-node K8s cluster is enough.

---

<a id="2"></a>
## 2 Meet the Agents
<img src="https://github.com/semework/AiMedOrchestra/blob/main/assets/router.png" style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 95%;"/> 

| ID | Service | Core Models | Input → Output |
|----|---------|-------------|----------------|
| **A1** | Data Synthesiser | CTGAN, VAE-Survival | schema → synthetic EHR rows |
| **A2** | Diagnostics LLM | FLAN-T5-XXL + rules | vitals + notes + findings → ranked diagnoses |
| **A3** | Imaging | ResNet-50 · EfficientNet-b3 · Grad-CAM | DICOM/PNG → label + heat-map |
| **A4** | Genomics | DNABERT · ESM-1b | VCF → ACMG class |
| **A5** | Drug-Discovery GNN | PyG GIN · GPT-GNN | target → 10 candidate molecules |
| **A6** | Treatment RL | PPO (sb3) | state → dose (“No/Low/Med/High”) |
| **A7** | Literature RAG | SBERT + T5 | query → summary + citations |
| **A8** | Trial Matcher | mini-BERT + regex | patient → recruiting trials |
| **A9** | Mental-Health Chat | DialoGPT - RoBERTa-sent | text → empathic reply + cue |
| **A10**| Ethics Auditor | fairness metrics + SHAP | logs → bias report |

---

<a id="3"></a>
## 3 Agent Details

### Data Synthesiser (A1)
Generates realistic synthetic EHR rows with **CTGAN**. Trains on tabular distributions, preserves utility while removing PHI — perfect for sandboxing downstream models.

---

### Diagnostics LLM (A2)
Fuses history, imaging and genomics into an AI-written differential. Feed a patient dict plus any free-text findings; returns ranked diagnoses and next-step recommendations in seconds.

---

### Imaging Agent (A3)
ResNet-50 + EfficientNet-b3 ensemble with Grad-CAM. Processes DICOM or PNG, outputs labels and heat-maps for fast visual verification.

---

### Genomics Classifier (A4)
Interprets VCF variants with DNABERT + ESM-1b, predicts ACMG pathogenicity and flags clinically actionable mutations for precision medicine.

---

### Drug-Discovery GNN (A5)
Graph-Neural pipeline (GIN, GPT-GNN) ranks top molecules for a disease target and explains reasoning in natural language.

---

### Treatment RL (A6)
Policy-gradient PPO trained in a simulated diabetes environment. Accepts raw severity or full patient profile; returns safe No/Low/Med/High dosing plans.

---

### Literature RAG (A7)
SBERT retrieves, T5 summarises → bulleted evidence with citations. Keeps recommendations anchored in current literature.

---

### Trial Matcher (A8)
Scrapes ClinicalTrials.gov recruiting studies, parses eligibility, filters with regex and a mini-BERT binary classifier. Outputs actionable NCT IDs + titles + links.

---

### Mental-Health Chat (A9)
DialoGPT dialogue plus Distil-RoBERTa emotion classifier. Returns empathetic text *and* a plain-English cue (e.g. “Possible anxiety symptoms”).

---

### Ethics & Bias Auditor (A10)
Simulates decisions on synthetic patients from A1, computes fairness metrics (demographic parity, equalised odds) and logs SHAP explanations for audit.

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

## 5 Installation & Quick Start

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
 
License
Released under the Business Source License 1.1 – free for non-commercial & research use; commercial use requires a license after the Change Date.
```
---

11 References
 
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