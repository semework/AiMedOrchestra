


<!-- CONTRIBUTING.md – AiMedOrchestra -->
# Contributing to **AiMedOrchestra**

First off, **thank you** for thinking about improving *AiMedOrchestra AI*!  
We welcome code, docs, models, benchmarks, and real-world feedback.  
*(Project license: **Business Source License 1.1** – see `LICENSE`)*

---

## 📜 Ground Rules
| Guideline | Why it matters |
|-----------|----------------|
| **Be kind & constructive.** | We follow the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) Code of Conduct. |
| **Dual-license consent.** | All contributions are dual-licensed under BSL-1.1 now and Apache-2.0 on the conversion date. By opening a PR you agree to this. |
| **No PHI / secrets.** | Never commit real patient data or private keys. Use synthetic samples under `data/`. |
| **Security first.** | Report vulnerabilities privately—see “Security Policy” below. |

---

## 🛠 Development Setup

```bash
git clone https://github.com/semework/AiMedOrchestra.git
cd AiMedOrchestra
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core runtime
pip install -e .[dev]                    # lint + test extras
pre-commit install                       # style hooks
pytest -q                                # unit tests
Tip: docker compose up --build will spin up every agent & the orchestrator in one go.