import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np




class GenomicsAgent:
    def __init__(self, model_name="dnabert-base", device=None):
        """
        Initialize the genomics agent with a pretrained transformer model
        :param model_name: HuggingFace pretrained model name (replace with a real variant pathogenicity model)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Known variants for quick lookup (extendable)
        self.KNOWN_PATHOGENIC = {
            "BRCA1 5382insC": "Pathogenic variant in BRCA1 (high risk for breast/ovarian cancer)",
            "TP53 missense mutation": "Pathogenic variant in TP53 (Li-Fraumeni syndrome)"
        }

        self.KNOWN_BENIGN = {
            "MTHFR C677T": "Likely benign variant, common in population"
        }

    def analyze(self, variant: str, sequence_context: str = None) -> str:
        """
        Analyze variant pathogenicity using pretrained model + heuristic + knowledge base

        :param variant: Variant string (e.g. "BRCA1 5382insC")
        :param sequence_context: Optional raw DNA sequence context surrounding variant
        :return: Interpretation string
        """
        variant_upper = variant.strip().upper()

        # 1. Check known variants first
        for key in self.KNOWN_PATHOGENIC:
            if key.upper() == variant_upper:
                return self.KNOWN_PATHOGENIC[key]

        for key in self.KNOWN_BENIGN:
            if key.upper() == variant_upper:
                return self.KNOWN_BENIGN[key]

        # 2. If sequence context is provided, run prediction
        if sequence_context:
            inputs = self.tokenizer(sequence_context, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()[0]

            # Assuming binary classification: 0 benign, 1 pathogenic
            pathogenic_prob = self.softmax(logits)[1]

            if pathogenic_prob > 0.7:
                return f"{variant}: Predicted Pathogenic (confidence: {pathogenic_prob:.2f})"
            elif pathogenic_prob < 0.3:
                return f"{variant}: Predicted Benign (confidence: {1 - pathogenic_prob:.2f})"
            else:
                return f"{variant}: Variant of uncertain significance (confidence: {pathogenic_prob:.2f})"

        # 3. Fallback heuristic if no sequence context
        if any(x in variant_upper for x in ["DEL", "FS", "X>", "STOP", "INS", "NONSENSE"]):
            return f"{variant}: Likely pathogenic (disruptive mutation detected)"
        elif any(x in variant_upper for x in ["MISSENSE", "SNP", "VARIANT"]):
            return f"{variant}: Variant of uncertain significance"
        else:
            return f"{variant}: Variant interpretation unknown (expert review needed)"

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

