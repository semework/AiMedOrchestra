# conversational_orchestrator.py
import re
from importlib import import_module
from pathlib import Path
from typing import Dict, Any

class ConversationalOrchestrator:
    """
    Conversational interface on top of aimedorchestraAgent.
    Understands simple natural commands and routes them to the right agent.
    """

    _BASE = "aimedorchestra.agents"

    _COMMANDS: Dict[str, str] = {
        r"(create|generate).*(patient|synthetic)": "data_synthesis",
        r"(diagnose|diagnosis|symptom)": "diagnostics",
        r"(diet|meal|nutrition)": "diet_planner",
        r"(drug|compound|molecule)": "drug_discovery",
        r"(ethic|bias|fairness)": "ethical_monitoring",
        r"(genom|variant|dna)": "genomics",
        r"(image|scan|mri|ct)": "imaging",
        r"(literature|paper|pubmed)": "literature_surveillance",
        r"(trial|clinical)": "clinical_trial_matching",
        r"(mental|psych|stress|worried)": "mental_health",
        r"(treat|therapy|plan)": "treatment_optimization",
    }

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def _get_agent(self, pkg: str):
        if pkg in self._cache:
            return self._cache[pkg]

        try:
            mod = import_module(f"{self._BASE}.{pkg}.agent")
            cls = next(obj for n, obj in vars(mod).items()
                       if n.lower().endswith("agent") and callable(obj))
            self._cache[pkg] = cls()
            return self._cache[pkg]
        except Exception as e:
            print(f"⚠️ Could not load {pkg}: {e}")
            return None

    def chat(self, user_text: str) -> str:
        """Natural language → agent call"""
        for pattern, pkg in self._COMMANDS.items():
            if re.search(pattern, user_text.lower()):
                agent = self._get_agent(pkg)
                if agent is None:
                    return f"(Sorry, {pkg} agent unavailable)"
                # assumes every agent implements `run` or equivalent
                if hasattr(agent, "run"):
                    return agent.run(user_text)
                elif hasattr(agent, "generate"):  # DataSynth special
                    num = int(re.search(r"\d+", user_text) or [1][0])
                    return agent.generate(num).to_dict("records")
                elif hasattr(agent, "plan"):
                    return agent.plan({"user": user_text})
                elif hasattr(agent, "diagnose"):
                    return agent.diagnose({"text": user_text})
                return f"(Agent {pkg} did not have a suitable method)"
        return "(I didn’t understand that request.)"
