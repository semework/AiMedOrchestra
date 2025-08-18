
# aimedorchestra/orchestrator/agent.py (crash‑resistant)
from __future__ import annotations

import os
import re
import json
import importlib.util
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

# -------------------- helpers --------------------

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20
}

def _spec_exists(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False

def _to_str(x: Any) -> str:
    try:
        if isinstance(x, str):
            return x
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        return str(x)

def _extract_int(text: Any, default: int = 1) -> int:
    s = _to_str(text)
    m = re.search(r"\b(\d+)\b", s)
    if m:
        try:
            return max(1, int(m.group(1)))
        except Exception:
            pass
    s_low = s.lower()
    for w, v in _NUMBER_WORDS.items():
        if re.search(rf"\b{w}\b", s_low):
            return max(1, v)
    return default

def _lower(s: Any) -> str:
    try:
        return str(s).lower().strip()
    except Exception:
        return ""

def _maybe_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(payload)

# -------------------- legacy pipeline --------------------
from aimedorchestra.agents.data_synthesis.agent import DataSynthesisAgent as DataSynthesizer
from aimedorchestra.agents.diagnostics.agent              import DiagnosticsAgent
from aimedorchestra.agents.imaging.agent                  import ImagingAgent
from aimedorchestra.agents.genomics.agent                 import GenomicsAgent
from aimedorchestra.agents.drug_discovery.agent           import DrugDiscoveryAgent
from aimedorchestra.agents.treatment_optimization.agent   import TreatmentAgent
from aimedorchestra.agents.literature_surveillance.agent  import LiteratureAgent
from aimedorchestra.agents.clinical_trial_matching.agent  import TrialMatchingAgent
from aimedorchestra.agents.mental_health.agent            import MentalHealthAgent
from aimedorchestra.agents.ethical_monitoring.agent       import EthicsAgent
from aimedorchestra.agents.diet_planner.agent             import DietPlannerAgent

class PipelineOrchestrator:
    def __init__(self):
        self.data_synth   = DataSynthesizer()
        self.diagnostics  = DiagnosticsAgent()
        self.imaging      = ImagingAgent()
        self.genomics     = GenomicsAgent()
        self.drug_disc    = DrugDiscoveryAgent()
        self.treatment    = TreatmentAgent()
        self.literature   = LiteratureAgent()
        self.trial_match  = TrialMatchingAgent()
        self.mental       = MentalHealthAgent()
        self.ethics       = EthicsAgent()
        self.diet         = DietPlannerAgent()

    def run_full_pipeline(self, patient: dict) -> dict:
        results                    = {}
        results["synthetic_row"]   = self.data_synth.generate(1).to_dict("records")[0]
        img_res                    = self.imaging.analyze("data/sample_image.jpg")
        gen_res                    = self.genomics.analyze_variant("BRCA1 5382insC")
        results["imaging"]         = img_res
        results["genomics"]        = gen_res
        results["diagnosis"]       = self.diagnostics.diagnose(patient)
        results["drug_suggestion"] = self.drug_disc.suggest(results["diagnosis"])
        results["diet_plan"]       = self.diet.plan(patient)
        results["treatment_plan"]  = self.treatment.plan(patient)
        results["literature"]      = self.literature.search(results["diagnosis"])
        results["trials"]          = self.trial_match.match(patient)[:5]
        results["mental_reply"]    = self.mental.respond("I feel worried …")
        results["ethics_report"]   = self.ethics.check(["dummy_log_entry_1", "dummy_log_entry_2"])
        return results

# -------------------- conversational router --------------------

INTENT_PATTERNS: Tuple[Tuple[str, str, str, str], ...] = (
    ("selftest", r"\b(self[-\s]?test|diagnostic\s+check|health\s+check)\b", "selftest", "selftest"),
    ("synthesize",       r"(create|generate|make).*(synthetic|patient|patients)", "data_synthesis", "generate"),
    ("synthesize_short", r"^(synthetic|patients?)\b",                            "data_synthesis", "generate"),
    ("diagnose", r"\b(diagnose|diagnosis|what\s+is\s+the\s+diagnosis)\b",        "diagnostics", "diagnose"),
    ("diet",     r"\b(diet|meal|nutrition|meal\s*plan|diet\s*plan)\b",           "diet_planner", "plan"),
    ("drug",     r"\b(drug|compound|molecule|lead\s*opt|hit\s*discovery)\b",     "drug_discovery", "suggest"),
    ("ethics",   r"\b(ethic|bias|fairness|explainability|audit)\b",              "ethical_monitoring", "check"),
    ("genomics", r"\b(genom|variant|snv|cnv|dna|gene)\b",                        "genomics", "analyze_variant"),
    ("imaging",  r"\b(image|imaging|scan|mri|ct|x-?ray|dicom)\b",                "imaging", "analyze"),
    ("literature", r"\b(literature|paper|pubmed|study|citation|references?)\b",  "literature_surveillance", "search"),
    ("trials",   r"\b(trial|clinical\s*trial|recruiting|nct)\b",                 "clinical_trial_matching", "match"),
    ("mental",   r"\b(mental|anxiety|depress|stress|worried|counsel)\b",         "mental_health", "respond"),
    ("treatment", r"\b(treat|therapy|protocol|care\s*plan|treatment\s*plan)\b",  "treatment_optimization", "plan"),
    ("pipeline",  r"\b(full\s*pipeline|run\s*all|end\s*to\s*end|orchestrate)\b", "orchestrator", "pipeline"),
)

class aimedorchestraAgent:
    _BASE_PKG = "aimedorchestra.agents"

    _KEYWORD_ROUTES: Dict[str, str] = {
        "trial": "clinical_trial_matching",
        "synthesis": "data_synthesis",
        "synthetic": "data_synthesis",
        "diagnostic": "diagnostics",
        "diagnose": "diagnostics",
        "diet": "diet_planner",
        "drug": "drug_discovery",
        "ethical": "ethical_monitoring",
        "genomic": "genomics",
        "variant": "genomics",
        "image": "imaging",
        "mri": "imaging",
        "ct": "imaging",
        "literature": "literature_surveillance",
        "mental": "mental_health",
        "treatment": "treatment_optimization",
    }

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._agents_present = self._discover_agents()

    # discovery without importing modules (prevents segfaults on bad native deps)
    def _discover_agents(self) -> Dict[str, bool]:
        found: Dict[str, bool] = {}
        for _, _, pkg, _ in INTENT_PATTERNS:
            if pkg in ("orchestrator", "selftest"):
                continue
            found[pkg] = _spec_exists(f"{self._BASE_PKG}.{pkg}.agent")
        for pkg in set(self._KEYWORD_ROUTES.values()):
            found.setdefault(pkg, _spec_exists(f"{self._BASE_PKG}.{pkg}.agent"))
        return found

    def _get_agent(self, pkg: str) -> Optional[Any]:
        if pkg in self._cache:
            return self._cache[pkg]
        try:
            mod = import_module(f"{self._BASE_PKG}.{pkg}.agent")
            cls = next(obj for name, obj in vars(mod).items()
                       if name.lower().endswith("agent") and callable(obj))
            self._cache[pkg] = cls()
            return self._cache[pkg]
        except Exception as e:
            print(f"⚠️  Failed to load {pkg}: {e}")
            return None

    def _parse_intent(self, text: Any) -> Tuple[str, str, str]:
        s = _lower(text)
        for name, pattern, pkg, handler in INTENT_PATTERNS:
            try:
                if re.search(pattern, s):
                    return (name, pkg, handler)
            except Exception:
                continue
        for kw, pkg in self._KEYWORD_ROUTES.items():
            if kw in s:
                return ("keyword", pkg, "auto")
        return ("", "", "")

    def chat(self, user_text: Any) -> str:
        return self.route(user_text)

    def route(self, user_text: Any) -> str:
        text = _to_str(user_text)
        intent, pkg, handler = self._parse_intent(text)

        if pkg == "selftest":
            return _maybe_json(self._run_self_tests())

        if pkg == "orchestrator" or intent == "pipeline":
            try:
                patient = self._maybe_extract_patient(text)
                res = PipelineOrchestrator().run_full_pipeline(patient)
                return _maybe_json(res)
            except Exception as e:
                return _maybe_json({"error": "pipeline_failed", "detail": str(e)})

        if not pkg:
            return _maybe_json({
                "message": "Try: 'create two synthetic patients', 'diagnose the patient with chest pain', "
                           "'find clinical trials for lung cancer', or 'run full pipeline on {\"age\":60, \"sex\":\"male\"}'."
            })

        if not self._agents_present.get(pkg, False):
            return _maybe_json({"error": "agent_unavailable", "agent": pkg})

        agent = self._get_agent(pkg)
        if agent is None:
            return _maybe_json({"error": "agent_import_failed", "agent": pkg})

        try:
            result = self._dispatch(agent, pkg, handler, text)
            return _maybe_json(result)
        except Exception as e:
            return _maybe_json({"error": "dispatch_failed", "agent": pkg, "detail": str(e)})

    def run_full_pipeline(self, patient: dict) -> dict:
        return PipelineOrchestrator().run_full_pipeline(patient)

    # dispatch with safety checks for risky deps
    def _dispatch(self, agent: Any, pkg: str, handler: str, text: str) -> Any:
        # Data synthesis
        if pkg == "data_synthesis":
            n = _extract_int(text, default=1)
            if hasattr(agent, "generate"):
                result = agent.generate(n)
                try:
                    import pandas as pd  # type: ignore
                    if isinstance(result, pd.DataFrame):
                        return result.to_dict("records")
                except Exception:
                    pass
                return result
            if hasattr(agent, "run"):
                return agent.run({"task": "synthesize", "n": n})
            return {"error": "data_synthesis: no entrypoint"}

        # Diagnostics
        if pkg == "diagnostics":
            patient = self._maybe_extract_patient(text)
            if hasattr(agent, "diagnose"):
                return agent.diagnose(patient)
            if hasattr(agent, "run"):
                return agent.run({"task": "diagnose", "patient": patient})
            return {"error": "diagnostics: no entrypoint"}

        # Diet
        if pkg == "diet_planner":
            patient = self._maybe_extract_patient(text)
            if hasattr(agent, "plan"):
                return agent.plan(patient)
            if hasattr(agent, "run"):
                return agent.run({"task": "diet", "patient": patient})
            return {"error": "diet_planner: no entrypoint"}

        # Drug discovery (guard RDKit)
        if pkg == "drug_discovery":
            if os.environ.get("AIMED_DISABLE_DRUG_DISCOVERY") == "1":
                return {"ok": False, "error": "drug_discovery disabled by env"}
            if not (_spec_exists("rdkit") or _spec_exists("rdkit.Chem")):
                return {"ok": False, "error": "rdkit_unavailable"}
            query = self._maybe_extract_query(text) or text
            if hasattr(agent, "suggest"):
                return agent.suggest(_to_str(query))
            if hasattr(agent, "run"):
                return agent.run({"task": "drug_suggest", "query": _to_str(query)})
            return {"error": "drug_discovery: no entrypoint"}

        # Ethics
        if pkg == "ethical_monitoring":
            logs = self._maybe_extract_logs(text)
            if hasattr(agent, "check"):
                return agent.check(logs)
            if hasattr(agent, "run"):
                return agent.run({"task": "ethics_check", "logs": logs})
            return {"error": "ethical_monitoring: no entrypoint"}

        # Genomics
        if pkg == "genomics":
            variant = self._maybe_extract_variant(text) or "BRCA1 5382insC"
            if hasattr(agent, "analyze_variant"):
                return agent.analyze_variant(_to_str(variant))
            if hasattr(agent, "run"):
                return agent.run({"task": "variant", "variant": _to_str(variant)})
            return {"error": "genomics: no entrypoint"}

        # Imaging (allow disable)
        if pkg == "imaging":
            if os.environ.get("AIMED_DISABLE_IMAGING") == "1":
                return {"ok": False, "error": "imaging disabled by env"}
            image = self._maybe_extract_image(text) or "data/sample_image.jpg"
            if hasattr(agent, "analyze"):
                return agent.analyze(_to_str(image))
            if hasattr(agent, "run"):
                return agent.run({"task": "imaging", "image": _to_str(image)})
            return {"error": "imaging: no entrypoint"}

        # Literature
        if pkg == "literature_surveillance":
            query = self._maybe_extract_query(text) or text
            if hasattr(agent, "search"):
                return agent.search(_to_str(query))
            if hasattr(agent, "run"):
                return agent.run({"task": "literature", "query": _to_str(query)})
            return {"error": "literature_surveillance: no entrypoint"}

        # Trials
        if pkg == "clinical_trial_matching":
            patient = self._maybe_extract_patient(text)
            if hasattr(agent, "match"):
                res = agent.match(patient)
                if isinstance(res, list):
                    return res[:5]
                return res
            if hasattr(agent, "run"):
                return agent.run({"task": "trial_match", "patient": patient})
            return {"error": "clinical_trial_matching: no entrypoint"}

        # Mental
        if pkg == "mental_health":
            msg = self._maybe_extract_message(text) or text
            if hasattr(agent, "respond"):
                return agent.respond(_to_str(msg))
            if hasattr(agent, "run"):
                return agent.run({"task": "mental_support", "message": _to_str(msg)})
            return {"error": "mental_health: no entrypoint"}

        # Fallback
        if hasattr(agent, "run"):
            return agent.run(_to_str(text))
        return {"error": f"Agent '{pkg}' has no recognized entrypoint."}

    # extractors
    def _maybe_extract_patient(self, text: Any) -> Dict[str, Any]:
        s = _to_str(text)
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        patient: Dict[str, Any] = {}
        low = s.lower()
        m = re.search(r"\bage\s*(\d{1,3})\b", low)
        if m: patient["age"] = int(m.group(1))
        else:
            m = re.search(r"\b(\d{1,3})\s*(yo|y/o|years?\s*old)\b", low)
            if m: patient["age"] = int(m.group(1))
        if re.search(r"\b(female|woman|she|her)\b", low): patient["sex"] = "female"
        elif re.search(r"\b(male|man|he|him)\b", low): patient["sex"] = "male"
        conds: List[str] = []
        if "diabet" in low: conds.append("diabetes")
        if "hypertens" in low or "high blood pressure" in low: conds.append("hypertension")
        if "asthma" in low: conds.append("asthma")
        if "cancer" in low: conds.append("cancer")
        if conds: patient["conditions"] = conds
        syms: List[str] = []
        for s_phrase in ("chest pain", "cough", "fever", "fatigue", "headache", "shortness of breath"):
            if s_phrase in low: syms.append(s_phrase)
        if syms: patient["symptoms"] = syms
        m = re.search(r"\b(in|near)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", _to_str(text))
        if m: patient["location"] = m.group(2)
        return patient or {"note": "minimal_patient_extracted_from_text"}

    def _maybe_extract_variant(self, text: Any) -> Optional[str]:
        s = _to_str(text)
        m = re.search(r"\b([A-Z0-9]{3,8}\s*[0-9A-Za-z\-\+_]+)\b", s)
        return m.group(1) if m else None

    def _maybe_extract_image(self, text: Any) -> Optional[str]:
        s = _to_str(text)
        m = re.search(r"([\w\-/\.]+(?:\.png|\.jpg|\.jpeg|\.dcm))", s, re.IGNORECASE)
        return m.group(1) if m else None

    def _maybe_extract_query(self, text: Any) -> Optional[str]:
        s = _to_str(text)
        m = re.search(r"\b(search|find|look\s*up)\b(.*)$", s, re.IGNORECASE)
        if m: return m.group(2).strip(" :,-")
        return None

    def _maybe_extract_logs(self, text: Any) -> Any:
        s = _to_str(text)
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return ["log_entry_1", "log_entry_2"]

    def _maybe_extract_message(self, text: Any) -> Optional[str]:
        return _to_str(text).strip() if text is not None else None

    # safer self‑test (skips risky agents by default)
    def _run_self_tests(self) -> Dict[str, Any]:
        tests = {
            "data_synthesis": "create 2 synthetic patients",
            "diagnostics":    "diagnose a 55 yo female with chest pain and cough",
            "diet_planner":   "diet plan for a 60 yo male with diabetes",
            # "drug_discovery": "drug suggestion for breast cancer",  # skipped by default
            "ethical_monitoring": "ethics check on [\"decisionA\",\"decisionB\"]",
            "genomics":       "analyze variant BRCA1 5382insC",
            # "imaging":        "analyze image data/sample_image.jpg",  # skipped by default
            "literature_surveillance": "search literature for lung cancer immunotherapy",
            "clinical_trial_matching": "find trials for a 65 yo male with lung cancer in Boston",
            "mental_health":  "I feel worried about my diagnosis",
            "treatment_optimization": "treatment plan for a 50 yo female with hypertension",
        }
        report: Dict[str, Any] = {}
        for pkg, cmd in tests.items():
            if not self._agents_present.get(pkg, False):
                report[pkg] = {"ok": False, "error": "agent_unavailable"}
                continue
            a = self._get_agent(pkg)
            if a is None:
                report[pkg] = {"ok": False, "error": "agent_import_failed"}
                continue
            try:
                out = self._dispatch(a, pkg, "auto", cmd)
                report[pkg] = {"ok": True, "result": out}
            except Exception as e:
                report[pkg] = {"ok": False, "error": str(e)}
        # annotate skips
        report["_note"] = "Self test skips 'drug_discovery' and 'imaging' by default. Set AIMED_DISABLE_*=0 and run them manually."
        return report
