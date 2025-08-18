
import os, json

# Keep it fast & stable by skipping heavy native deps by default
os.environ.setdefault("AIMED_DISABLE_DRUG_DISCOVERY", "1")
os.environ.setdefault("AIMED_DISABLE_IMAGING", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from aimedorchestra.orchestrator.agent import aimedorchestraAgent
import gradio as gr

orch = aimedorchestraAgent()

def chat_fn(message, history):
    try:
        reply = orch.route(message)
        # Pretty-print JSON if that's what we got
        try:
            obj = json.loads(reply)
            reply = json.dumps(obj, indent=2)
        except Exception:
            pass
        return reply
    except Exception as e:
        return f"Error: {e}"

gr.ChatInterface(
    fn=chat_fn,
    title="AiMedOrchestra — Quick Chat",
    description="Type natural requests (e.g., 'create two synthetic patients', 'diagnose a 55 yo female with chest pain')."
).launch()
