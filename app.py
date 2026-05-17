import streamlit as st

from utils import extract_text
from rag_pipeline import (
    create_vector_store,
    ask_question
)

st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="📄"
)

st.title("📄 AI Document Support Chatbot")

# -----------------
# Session State
# -----------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# -----------------
# Upload Document
# -----------------

uploaded_files = st.file_uploader(
    "Upload PDF / TXT / DOCX",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

# Process ONLY ONCE
if (
    uploaded_files
    and not st.session_state.document_loaded
):

    with st.spinner(
        "Processing document..."
    ):

        full_text = ""

        for file in uploaded_files:
            full_text += extract_text(file)

        st.session_state.vector_store = (
            create_vector_store(full_text)
        )

        st.session_state.document_loaded = True

    st.success(
        "Document Ready!"
    )

# -----------------
# Question
# -----------------

question = st.text_input(
    "Ask question:"
)

if st.button("Send"):

    if not question:

        st.warning(
            "Type question first"
        )

    elif (
        st.session_state.vector_store
        is None
    ):

        st.warning(
            "Upload document first"
        )

    else:

        with st.spinner(
            "Thinking..."
        ):

            response = ask_question(
                st.session_state.vector_store,
                question
            )

        st.write("### Answer")
        st.write(response)