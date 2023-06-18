from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
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
import re

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def pdf_loader(pdf) -> str:
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def create_db_from_pdf_text(text: str) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )
    docs = text_splitter.split_text(text)
    db = FAISS.from_texts(docs, embedding=embeddings)
    return db


def search_docs(db: VectorStore, query: str, k=2) -> List[Document]:
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    return docs_page_content


def get_response_from_query(
    docs_page_content: List[Document], query: str
) -> Dict[str, Any]:
    llm = OpenAI()

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
    chain = LLMChain(llm=llm, prompt=prompt)

    answer = chain(
        {"input_documents": docs_page_content, "question": query},
        return_only_outputs=True,
    )
    return answer


def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs
