"""
streamlit_app.py

Web UI for the Context-Aware Customer Support RAG Bot, built for free
hosting on Streamlit Community Cloud (share.streamlit.io).

Local run:
    streamlit run streamlit_app.py

Deployed: see README.md "Deploying a Free Hosted Link" section.
"""

import streamlit as st

from rag_pipeline import (
    ConfigurationError,
    answer_query,
    ensure_setup,
    get_groq_client,
    load_vector_store,
)

st.set_page_config(page_title="NimbusCart Support Assistant", page_icon="💬")

st.title("💬 NimbusCart Customer Support Assistant")
st.caption(
    "Context-Aware Customer Support RAG Bot — answers are generated only "
    "from the company FAQ knowledge base, personalized to your membership tier."
)

with st.expander("Try a sample user_id"):
    st.markdown(
        "- **101** — Riya Sharma (Gold)\n"
        "- **102** — Aman Verma (Silver)\n"
        "- **103** — Neha Iyer (Platinum)\n"
        "- **999** — not in the database (tests the error path)"
    )


@st.cache_resource(show_spinner="Setting up database and knowledge base (first run only)...")
def bootstrap():
    ensure_setup()
    return True


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_vector_store():
    return load_vector_store()


@st.cache_resource(show_spinner=False)
def get_client():
    return get_groq_client()


# Set up the pipeline once per session, surfacing config errors clearly
# instead of letting the app crash.
setup_error = None
vector_store = None
client = None
try:
    bootstrap()
    vector_store = get_vector_store()
    client = get_client()
except ConfigurationError as exc:
    setup_error = str(exc)
except Exception as exc:
    setup_error = f"Unexpected setup error: {exc}"

if setup_error:
    st.error(
        f"Setup error: {setup_error}\n\n"
        "If you're the app owner: for a deployed app, add GROQ_API_KEY under "
        "'Secrets' in your hosting platform's settings. Also make sure "
        "users.db and faiss_index/ were generated before deploying."
    )
    st.stop()

col1, col2 = st.columns([1, 3])
with col1:
    user_id_input = st.text_input("user_id", value="101")
with col2:
    query_input = st.text_input("Your question", value="What is the refund policy?")

if st.button("Ask", type="primary"):
    try:
        user_id = int(user_id_input.strip())
    except ValueError:
        st.warning("User not found. Please enter a valid user_id.")
    else:
        if not query_input.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                answer = answer_query(vector_store, client, user_id, query_input.strip())
            st.markdown("### Answer")
            st.write(answer)
