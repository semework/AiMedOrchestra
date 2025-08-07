# aimedorchestraagent.py  (save/overwrite)
from importlib import import_module
from pathlib import Path
from typing import Dict, Callable

class aimedorchestraAgent:
    """
    Lazy router: real sub-agents are only imported *and* instantiated
    the first time their keyword bucket is triggered.
    """

    _BASE = "aimedorchestra.agents"
    _ROUTES: Dict[str, str] = {
        "trial":            "clinical_trial_matching",
        "synthesis":        "data_synthesis",
        "diagnostic":       "diagnostics",
        "diet":             "diet_planner",
        "drug":             "drug_discovery",
        "ethical":          "ethical_monitoring",
        "genomic":          "genomics",
        "image":            "imaging",
        "literature":       "literature_surveillance",
        "mental":           "mental_health",
        "treatment":        "treatment_optimization",
    }

    def __init__(self):
        self._cache: Dict[str, Callable[[str], str]] = {}  # pkg → instance

    # -----------------------------------------------------------
    def _get_agent(self, pkg: str):
        """Import + construct the sub-agent on first use."""
        if pkg in self._cache:
            return self._cache[pkg]

        try:
            mod = import_module(f"{self._BASE}.{pkg}.agent")
            cls = next(obj for n, obj in vars(mod).items()
                       if n.lower().endswith("agent") and callable(obj))
            self._cache[pkg] = cls()        # heavy work happens *here*
            return self._cache[pkg]
        except Exception as e:
            print(f"⚠️  Failed to load {pkg}: {e}")
            self._cache[pkg] = lambda *_: "(agent unavailable)"
            return self._cache[pkg]

    # -----------------------------------------------------------
    def route(self, user_text: str) -> str:
        txt = user_text.lower()
        for kw, pkg in self._ROUTES.items():
            if kw in txt:
                agent = self._get_agent(pkg)
                return agent.run(user_text)  # type: ignore[attr-defined]
        # fallback orchestrator (lazy, too)
        from aimedorchestra.orchestrator.agent import aimedorchestraAgent as Orch
        return Orch().run(user_text)        # type: ignore[attr-defined]
