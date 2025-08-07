# diagnostics_agent.py
# ---------------------------------------------------------------
# pip install transformers torch --upgrade
# ---------------------------------------------------------------
from transformers import pipeline


class DiagnosticsAgent:
    """
    DiagnosticsAgent
    ----------------
    • Text-generation backbone: GPT-2 (swap in a larger model if desired)
    • Inputs:
        patient  : dict  – age, sex, known conditions, meds, etc.
        imaging  : str   – radiology or pathology report (optional)
        genomics : str   – variant list / polygenic score summary (optional)
    • Output: single integrated differential-diagnosis paragraph.
    """

    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model="gpt2",
            max_length=256,
            do_sample=True,
            temperature=0.7,
        )

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------
    def diagnose(self, patient, imaging: str | None = None, genomics: str | None = None) -> str:
        """
        Creates a differential diagnosis combining patient history,
        imaging findings, and genomic results.

        Parameters
        ----------
        patient   : dict   e.g. {"age": 50, "sex": "M", "conditions": ["Diabetes"]}
        imaging   : str    free-text imaging report                      (optional)
        genomics  : str    free-text genomic summary / variant list      (optional)

        Returns
        -------
        str – a coherent diagnostic note (first ~1–2 paragraphs).
        """
        # --- Build prompt ----------------------------------------------------
        prompt = [
            "You are an experienced clinician writing a concise diagnostic impression.",
            f"Patient information: {patient}.",
        ]

        if imaging:
            prompt.append(f"Imaging findings: {imaging}.")
        if genomics:
            prompt.append(f"Genomic results: {genomics}.")
        prompt.append(
            "Based on the above, provide the top probable diagnoses (with brief rationale) "
            "and suggest next steps for confirmation or management."
        )

        prompt = " ".join(prompt)

        # --- Generate response ----------------------------------------------
        generated = self.generator(prompt, num_return_sequences=1)[0]["generated_text"]
        # Strip the prompt from the front to return only the AI’s answer
        return generated[len(prompt):].strip()


# ---------------------------- smoke test ----------------------------
if __name__ == "__main__":
    agent = DiagnosticsAgent()
    result = agent.diagnose(
        {"age": 50, "conditions": ["Diabetes"]},
        imaging="CT abdomen shows mild fatty liver; no focal lesions.",
        genomics="HNF1A p.Gly31Ala variant of uncertain significance."
    )
    print(result)
