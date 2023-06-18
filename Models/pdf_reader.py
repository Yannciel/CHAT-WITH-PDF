from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import find_dotenv, load_dotenv
import re

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def pdf_loader(pdf) -> str:
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        for page in pdf.pages:
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
