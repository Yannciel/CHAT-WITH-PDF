from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from typing import Any, Dict, List
from langchain.vectorstores import VectorStore
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import streamlit as st

load_dotenv(find_dotenv())


@st.cache(allow_output_mutation=True)
def search_docs(db: VectorStore, query: str, k=2) -> List[Document]:
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    return docs_page_content


@st.cache(allow_output_mutation=True)
def get_response_from_query(
    docs_page_content: List[Document], query: str
) -> Dict[str, Any]:
    llm = OpenAI("gpt-3.5-turbo-16k")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
            Vous êtes un assistant utile capable de répondre aux questions sur les textes de pdf en se basant sur la contenu du pdf fourni.
            Inclure toujours une section "Sources" dans votre réponse, comprenant uniquement l'ensemble minimal des sources nécessaires pour répondre à la question.

            Répondez à la question suivante : {question}
            En recherchant dans les textes de pdf suivant : {docs}

            Utilisez uniquement les informations factuelles de la textes pour répondre à la question.

            Si vous estimez ne pas avoir suffisamment d'informations pour répondre à la question, dites "Je ne sais pas".

            Vos réponses doivent être détaillées et complètes.
        """,
    )
    chain = load_qa_with_sources_chain(prompt=prompt)

    answer = chain(
        {"input_documents": docs_page_content, "question": query},
        return_only_outputs=True,
    )
    return answer


@st.cache(allow_output_mutation=True)
def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs
