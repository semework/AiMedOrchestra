# mental_health_agent.py
# -------------------------------------------------------------------------
# pip install transformers torch --upgrade
# (model downloads happen automatically on first run)
# -------------------------------------------------------------------------
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


class MentalHealthAgent:
    """
    • Dialog engine: DialoGPT-medium (conversational replies)
    • Emotion detector: DistilRoBERTa fine-tuned on GoEmotions
      ─ labels: anger, disgust, fear, joy, neutral, sadness, surprise
    • Output: chatbot reply  +  plain-English diagnosis
    """

    _EMOTION2DX = {
        "anger":    "Possible anger / frustration",
        "disgust":  "Possible aversion / distress",
        "fear":     "Possible anxiety symptoms",
        "sadness":  "Possible depressive symptoms",
        "joy":      "Positive mood",
        "surprise": "Heightened arousal / surprise",
        "neutral":  "No strong emotional distress detected",
    }

    def __init__(self, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Load models ──────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.dialog_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium"
        ).to(self.device)

        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,           # returns list of all labels with scores
            device=0 if self.device.type == "cuda" else -1,
        )

        self.chat_history_ids = None

    # ────────────────────────────────────────────────────────────────
    # Conversation + diagnosis
    # ────────────────────────────────────────────────────────────────
    def respond(self, msg: str) -> dict:
        """
        Returns
        -------
        {
          "reply": "chatbot response …",
          "diagnosis": "Possible anxiety symptoms",
          "emotion": "fear",
          "confidence": 0.89
        }
        """
        # ---------- DialoGPT reply ----------
        new_ids = self.tokenizer.encode(msg + self.tokenizer.eos_token,
                                        return_tensors="pt").to(self.device)

        bot_input = (
            torch.cat([self.chat_history_ids, new_ids], dim=-1)
            if self.chat_history_ids is not None else new_ids
        )

        self.chat_history_ids = self.dialog_model.generate(
            bot_input, max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        reply = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input.shape[-1]:][0],
            skip_special_tokens=True,
        )

        # ---------- Emotion → diagnosis ----------
        emo_scores = self.emotion_classifier(msg)[0]        # list of dicts
        top = max(emo_scores, key=lambda x: x["score"])
        emotion      = top["label"]
        confidence   = round(top["score"], 2)
        diagnosis    = self._EMOTION2DX.get(emotion, "Unable to determine")

        # Optionally append empathy
        if emotion in ("fear", "sadness", "anger"):
            reply += " I’m sorry you’re feeling this way. You’re not alone—would you like some coping strategies or professional resources?"

        return {
            "reply": reply,
            "diagnosis": diagnosis,
            "emotion": emotion,
            "confidence": confidence,
        }


# ----------------- simple test -----------------
if __name__ == "__main__":
    agent = MentalHealthAgent()
    out = agent.respond("I feel anxious today.")
    print(out)
