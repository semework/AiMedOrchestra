# aimedorchestra/orchestrator/agent.py
"""
aimedorchestraAgent
‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒
A very light orchestrator that:
1. Holds references to every individual agent.
2. Exposes one helper,  run_full_pipeline(patient),
   which calls each agent’s key method in sequence.
Replace / expand however you like.
"""

# ⬇️  renamed class (keeps old alias)
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


class aimedorchestraAgent:
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

    # ---------------------------------------------------------------
    def run_full_pipeline(self, patient: dict) -> dict:
        """One-shot demo of every agent cooperating on a patient dict."""
        results                    = {}
        results["synthetic_row"]   = self.data_synth.generate(1).to_dict("records")[0]

        img_res                    = self.imaging.analyze("data/sample_image.jpg")
        gen_res                    = self.genomics.analyze_variant("BRCA1 5382insC")
        results["imaging"]         = img_res
        results["genomics"]        = gen_res

        results["diagnosis"]       = self.diagnostics.diagnose(
            patient, img_res, gen_res
        )
        results["drug_suggestion"] = self.drug_disc.suggest(results["diagnosis"])
        results["diet_plan"]       = self.diet.plan(patient)
        results["treatment_plan"]  = self.treatment.plan(patient)
        results["literature"]      = self.literature.search(results["diagnosis"])
        results["trials"]          = self.trial_match.match(patient)[:5]
        results["mental_reply"]    = self.mental.respond("I feel worried …")
        results["ethics_report"]   = self.ethics.check(
            ["dummy_log_entry_1", "dummy_log_entry_2"]
        )
        return results
