
import json, os, time, traceback
import streamlit as st

# Lower thread use to reduce native lib contention
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")      # helps some MKL/OMP clashes
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")       # prevent Qt GUI issues
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")  # OpenCV sometimes segfaults with OpenCL

try:
    from aimedorchestra.orchestrator.agent import aimedorchestraAgent
except Exception as e:
    st.error("Could not import package `aimedorchestra`. Did you run `pip install -e .` in the repo root?")
    st.stop()

st.set_page_config(page_title="AiMedOrchestra — Chat", page_icon="🧠", layout="wide")

left, right = st.columns([0.08, 0.92])
with left:
    try:
        st.image("assets/AiMed_logo.png", width=48)
    except Exception:
        pass
with right:
    st.title("AiMedOrchestra — Conversational Chat")

st.caption("Try: **create two synthetic patients**, **diagnose a 55 yo female with chest pain**, **find clinical trials for lung cancer near Boston**.")

with st.sidebar:
    st.header("Tools")
    safe_test = st.button("Run Self Test (safe)")
    risky = st.checkbox("Include risky agents (RDKit/OpenCV)", value=False)
    if risky:
        st.caption("Will also test Drug Discovery (RDKit) and Imaging (OpenCV/PyDICOM). May segfault on some systems.")

if "messages" not in st.session_state:
    st.session_state.messages = []

orch = aimedorchestraAgent()

# Self tests
if safe_test:
    try:
        # default route('self test') already skips risky agents per orchestrator code
        report = orch.route("self test")
        st.sidebar.success("Self test complete.")
        st.sidebar.code(report, language="json")
        if risky:
            # Manually test risky ones
            risky_out = {}
            if not os.environ.get("AIMED_DISABLE_DRUG_DISCOVERY") == "1":
                risky_out["drug_discovery"] = orch.route("drug suggestion for breast cancer")
            if not os.environ.get("AIMED_DISABLE_IMAGING") == "1":
                risky_out["imaging"] = orch.route("analyze image data/sample_image.jpg")
            st.sidebar.code(json.dumps({"risky": risky_out}, indent=2), language="json")
    except Exception as e:
        st.sidebar.error(f"Self test error: {e}")
        st.sidebar.text(traceback.format_exc())

# History
for role, content in st.session_state.messages:
    with st.chat_message(role):
        if role == "assistant":
            try:
                obj = json.loads(content)
                st.code(json.dumps(obj, indent=2), language="json")
            except Exception:
                st.write(content)
        else:
            st.write(content)

prompt = st.chat_input("Your message")
if prompt:
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)
    try:
        reply = orch.route(prompt)
    except Exception as e:
        reply = json.dumps({"error": str(e)}, indent=2)
    st.session_state.messages.append(("assistant", reply))
    with st.chat_message("assistant"):
        try:
            obj = json.loads(reply)
            st.code(json.dumps(obj, indent=2), language="json")
        except Exception:
            st.write(reply)
