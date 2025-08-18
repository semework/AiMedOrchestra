
import json
import streamlit as st

# Canonical import from the package
try:
    from aimedorchestra.orchestrator.agent import aimedorchestraAgent
except Exception as e:
    st.error("Could not import package `aimedorchestra`. Did you run `pip install -e .` in the repo root?")
    st.stop()

st.set_page_config(page_title="AiMedOrchestra — Chat", page_icon="🧠", layout="wide")

# Header with small logo
left, right = st.columns([0.08, 0.92])
with left:
    try:
        st.image("assets/AiMed_logo.png", width=48)
    except Exception:
        pass
with right:
    st.title("AiMedOrchestra — Conversational Chat")

st.caption("Type natural requests like **create two synthetic patients**, **diagnose a 55 yo female with chest pain**, or **find clinical trials for lung cancer near Boston**.")

# Sidebar: tools
with st.sidebar:
    st.header("Tools")
    run_selftest = st.button("Run Self Test")
    st.caption("Runs a quick smoke test across available agents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

orch = aimedorchestraAgent()

# Self test handling
if run_selftest:
    with st.spinner("Running self test..."):
        try:
            report = orch.route("self test")
        except Exception as e:
            report = json.dumps({"error": str(e)}, indent=2)
    st.sidebar.success("Self test complete.")
    st.sidebar.code(report, language="json")

# Display history
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

# Chat input
prompt = st.chat_input("Your message")
if prompt:
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # Orchestrator reply
    try:
        reply = orch.route(prompt)  # returns JSON string (usually)
    except Exception as e:
        reply = json.dumps({"error": str(e)}, indent=2)

    st.session_state.messages.append(("assistant", reply))
    with st.chat_message("assistant"):
        try:
            obj = json.loads(reply)
            st.code(json.dumps(obj, indent=2), language="json")
        except Exception:
            st.write(reply)
