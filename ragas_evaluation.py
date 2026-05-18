import os
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def evaluate_response(question, answer, contexts):
    try:
        # ✅ Use Groq instead of OpenAI
        groq_llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

        # ✅ Use HuggingFace embeddings instead of OpenAI
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ✅ Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(groq_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

        # ✅ Set up metrics with custom LLM + embeddings
        faithfulness = Faithfulness(
            llm=ragas_llm
        )
        answer_relevancy = AnswerRelevancy(
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy]
        )

        scores = result.to_pandas()

        faithfulness_score = round(
            float(scores["faithfulness"].iloc[0]), 2
        )
        relevancy_score = round(
            float(scores["answer_relevancy"].iloc[0]), 2
        )

        return {
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "error": None
        }

    except Exception as e:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "error": str(e)
        }