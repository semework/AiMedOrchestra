# aimedorchestra/agents/clinical_trial_matching/agent.py

from __future__ import annotations
import json
import os
from difflib import SequenceMatcher
from typing import List, Dict, Any

def _norm(s: str) -> str:
    return " ".join(str(s).lower().strip().split())

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

class TrialMatchingAgent:
    """
    Offline trial matcher.
    - Loads trials from data/trials.json (schema below) or uses built-in fallback.
    - Patient input: {"age": 50, "conditions": ["Diabetes", ...]}
    - Returns: list of trial dicts with keys:
        id, title, condition, min_age, max_age, summary, locations
    """

    def __init__(self, trials_path: str = "data/trials.json", min_similarity: float = 0.6):
        self.trials_path = trials_path
        self.min_similarity = float(min_similarity)
        self.trials: List[Dict[str, Any]] = self._load_trials(trials_path)
        if not self.trials:
            self.trials = self._fallback_trials()

        # normalize and ensure required keys exist
        for t in self.trials:
            t.setdefault("summary", "")
            t.setdefault("locations", [])
            t["condition_norm"] = _norm(t.get("condition", ""))

    def _load_trials(self, path: str) -> List[Dict[str, Any]]:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "trials" in data:
                    data = data["trials"]
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[TrialMatchingAgent] Failed to load {path}: {e}")
        return []

    def _fallback_trials(self) -> List[Dict[str, Any]]:
        # Minimal, realistic examples (kept small for demo stability)
        return [
            {
                "id": "NCT-DM-0001",
                "title": "Phase 3: Metformin vs. SGLT2 add-on for Type 2 Diabetes",
                "condition": "Diabetes",
                "min_age": 18, "max_age": 75,
                "summary": "Compares glycemic control and safety outcomes for T2D adults on metformin with/without SGLT2 inhibitor.",
                "locations": ["Boston, MA", "Chicago, IL"]
            },
            {
                "id": "NCT-HTN-0002",
                "title": "Lifestyle/DASH vs ARB monotherapy in Stage 1 Hypertension",
                "condition": "Hypertension",
                "min_age": 30, "max_age": 70,
                "summary": "Evaluates BP reduction with DASH diet + exercise vs ARB alone over 24 weeks.",
                "locations": ["Dallas, TX", "Seattle, WA"]
            },
            {
                "id": "NCT-DM-0003",
                "title": "GLP-1 RA adjunct therapy for Type 2 Diabetes with obesity",
                "condition": "Diabetes Mellitus",
                "min_age": 18, "max_age": 80,
                "summary": "Assesses weight and A1c change when adding GLP-1 RA to standard of care.",
                "locations": ["San Diego, CA"]
            },
            {
                "id": "NCT-MIX-0004",
                "title": "Cardio-metabolic outcomes in mixed metabolic syndrome",
                "condition": "Metabolic Syndrome",
                "min_age": 40, "max_age": 85,
                "summary": "Longitudinal outcomes registry for patients with DM/HTN overlap.",
                "locations": ["Remote"]
            }
        ]

    def match(self, patient: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        age = patient.get("age", None)
        conds = patient.get("conditions", [])
        if isinstance(conds, str):
            conds = [conds]
        conds_norm = [_norm(c) for c in conds if str(c).strip()]

        ranked: List[tuple[float, Dict[str, Any]]] = []

        for t in self.trials:
            # Age window check
            if age is not None:
                if not (t.get("min_age", 0) <= age <= t.get("max_age", 200)):
                    continue

            # Condition similarity: require each patient condition to match the trial condition reasonably
            cond_score = 1.0
            for pc in conds_norm:
                sim = _similar(pc, t["condition_norm"])
                cond_score = min(cond_score, sim)
            if conds_norm and cond_score < self.min_similarity:
                continue

            # Simple score: prioritize condition similarity and narrower age ranges
            age_span = max(1, (t.get("max_age", 200) - t.get("min_age", 0)))
            score = cond_score + 0.1 * (1.0 / age_span)

            ranked.append((score, t))

        ranked.sort(key=lambda x: x[0], reverse=True)
        results = [dict(r[1]) for r in ranked[:top_k]]

        # drop helper key used for matching
        for r in results:
            r.pop("condition_norm", None)
        return results

    # ------------------------------------------------------------------
    # Convenience entry-point so routers can call .run(text)
    # ------------------------------------------------------------------
    def run(self, query: str) -> str:
        """
        Accepts a free-text prompt, extracts basic condition keywords,
        calls .match(), and returns a short, human-readable summary.
        """
        conds = []
        for word in ("diabetes", "hypertension", "cancer", "stroke"):
            if word in query.lower():
                conds.append(word.title())
        patient = {"age": 50, "conditions": conds or ["General"]}
        hits = self.match(patient)[:3]
        if not hits:
            return "No recruiting trials found."
        bullet = "\n".join(f"- {t['id']}: {t['title']}" for t in hits)
        return f"Top recruiting trials:\\n{bullet}"

