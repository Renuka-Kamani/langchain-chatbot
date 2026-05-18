import streamlit as st
from utils import extract_text
from rag_pipeline import create_vector_store, ask_question
from ragas_evaluation import evaluate_response

st.set_page_config(
    page_title="Universal AI Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Universal Document & Chat Assistant")
st.write("Chat normally or upload documents.")

# -----------------------
# Session State
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# -----------------------
# Sidebar Upload
# -----------------------

with st.sidebar:
    st.header("📁 Document Center")

    uploaded_files = st.file_uploader(
        "Upload Files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        current_files = [f.name for f in uploaded_files]

        if current_files != st.session_state.processed_files:
            with st.spinner("Processing documents..."):
                full_text = ""
                for file in uploaded_files:
                    full_text += extract_text(file)

                st.session_state.vector_store = create_vector_store(full_text)
                st.session_state.processed_files = current_files

            st.success("Documents Ready!")

    # ✅ RAGAS info in sidebar
    if st.session_state.vector_store:
        st.divider()
        st.markdown("### 📊 RAGAS Evaluation")
        st.info(
            "Every answer is auto-evaluated.\n\n"
            "**Faithfulness**: Is the answer grounded in the document?\n\n"
            "**Answer Relevancy**: Does the answer address the question?"
        )

# -----------------------
# Chat History
# -----------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # ✅ Show RAGAS scores if they exist
        if (
            message["role"] == "assistant"
            and "scores" in message
            and message["scores"]
        ):
            scores = message["scores"]
            if scores["error"] is None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "🎯 Faithfulness",
                        f"{scores['faithfulness']:.0%}"
                    )
                with col2:
                    st.metric(
                        "💡 Answer Relevancy",
                        f"{scores['answer_relevancy']:.0%}"
                    )

# -----------------------
# Chat Input
# -----------------------

prompt = st.chat_input("Type message...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # ✅ Get answer + contexts
                response, contexts = ask_question(
                    st.session_state.vector_store,
                    prompt
                )

                st.write(response)

                scores = None

                # ✅ Only evaluate if document is uploaded
                if (
                    st.session_state.vector_store is not None
                    and contexts
                ):
                    with st.spinner("Evaluating with RAGAS..."):
                        scores = evaluate_response(
                            question=prompt,
                            answer=response,
                            contexts=contexts
                        )

                    if scores and scores["error"] is None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "🎯 Faithfulness",
                                f"{scores['faithfulness']:.0%}"
                            )
                        with col2:
                            st.metric(
                                "💡 Answer Relevancy",
                                f"{scores['answer_relevancy']:.0%}"
                            )
                    elif scores and scores["error"]:
                        st.warning(
                            f"RAGAS evaluation failed: {scores['error']}"
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "scores": scores
                })

            except Exception as e:
                st.error(str(e))