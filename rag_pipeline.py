import os

from dotenv import load_dotenv

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_groq import (
    ChatGroq
)

load_dotenv()


# ------------------------
# Create Vector Store
# ------------------------

def create_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    return vector_store


# ------------------------
# Ask Question
# ------------------------

def ask_question(
    vector_store,
    question
):

    docs = vector_store.similarity_search(
        question,
        k=6
    )

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    llm = ChatGroq(
        groq_api_key=os.getenv(
            "GROQ_API_KEY"
        ),
        model_name=
        "llama-3.1-8b-instant"
    )

    prompt = f"""
You are an intelligent
document assistant.

Rules:
1. Answer ONLY using
the provided document.

2. If answer is not in
document, say:

"I could not find this
information in the document."

3. Give accurate and
short answers.

DOCUMENT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(
        prompt
    )

    return response.content