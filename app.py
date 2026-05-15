import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Streamlit settings
st.set_page_config(page_title="AI RAG Chatbot")

st.title("🤖 AI RAG Chatbot using LangChain")

with st.sidebar:
    st.header("📌 About")
    st.write("🤖 AI RAG Chatbot")
    st.write("Built using:")
    st.write("- Gemini 2.5 Flash")
    st.write("- LangChain")
    st.write("- FAISS Vector Database")
    st.write("- Streamlit")

# Check API key
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Missing GOOGLE_API_KEY in .env file")
    st.stop()

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Show old messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF (Optional)",
    type="pdf"
)

# Process PDF button
if uploaded_file and st.button("Process PDF"):

    with st.spinner("Processing PDF..."):

        # Save file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        # Create vector DB
        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        # Save in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_processed = True

    st.success("PDF processed successfully!")

# Chat input
question = st.chat_input("Ask anything...")

if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    try:

        # If PDF exists → RAG mode
        if st.session_state.pdf_processed:

            relevant_docs = (
                st.session_state.vectorstore.similarity_search(
                    question,
                    k=3
                )
            )

            context = "\n\n".join(
                [doc.page_content for doc in relevant_docs]
            )

            prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the PDF context.

If answer is not found in PDF say:
"I could not find this information in the PDF."

Context:
{context}

Question:
{question}
"""

            response = llm.invoke(prompt)

        # Normal chat
        else:

            response = llm.invoke(question)

        answer = response.content

    except Exception as e:

        answer = f"Error: {e}"

    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)