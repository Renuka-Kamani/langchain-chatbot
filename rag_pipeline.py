import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from prompts import SYSTEM_PROMPT

load_dotenv()


def create_vector_store(text):
    if not text.strip():
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


def ask_question(vector_store, question):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    if vector_store is not None:
        docs = vector_store.similarity_search(question, k=4)

        # ✅ Return contexts as a list for RAGAS
        contexts = [doc.page_content for doc in docs]
        context_str = "\n\n".join(contexts)

        prompt = f"""
{SYSTEM_PROMPT}

Document Context:
{context_str}

User Question:
{question}

Answer:
"""
        response = llm.invoke(prompt)
        return response.content, contexts  # ✅ return both

    else:
        prompt = f"""
{SYSTEM_PROMPT}

The user has not uploaded a document.

Question:
{question}

Answer naturally.
"""
        response = llm.invoke(prompt)
        return response.content, []  # ✅ empty contexts for normal chat