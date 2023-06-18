import streamlit as st
import pickle
from openai.error import OpenAIError
from langchain.callbacks import get_openai_callback
from utils import (
    pdf_loader,
    create_db_from_pdf_text,
    search_docs,
    get_response_from_query,
)
import os


# Sidebar contents
with st.sidebar:
    st.title("Chat with PDF💬")
    st.markdown(
        """
    ## About
    This is a demo of chat with pdf files by using:
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    """
    )


def clear_submit():
    st.session_state["submit"] = False


def get_embeddings(pdf: str):
    store_name = pdf.name[:-4]
    store_name_embeddings = (
        "/Users/mingqiangwang/Desktop/Codes/CHAT-WITH-PDF/Docs/" + store_name
    )
    if os.path.exists("{}.pkl".format(store_name_embeddings)):
        with open("{}.pkl".format(store_name_embeddings), "rb") as f:
            VectorStored = pickle.load(f)
        print("Embedding loaded from the local disk")
    else:
        text = pdf_loader(pdf)
        VectorStored = create_db_from_pdf_text(text)
        with open("{}.pkl".format(store_name_embeddings), "wb") as f:
            pickle.dump(VectorStored, f)
    return store_name, VectorStored


def main():
    st.header("Chat with PDF 💬")
    # upload a PDF file
    pdf = st.file_uploader(
        "Upload your PDF",
        type="pdf",
        on_change=clear_submit,
    )

    # st.write(pdf)
    uploaded_file = None
    if pdf is not None:
        # # embeddings
        store_name, VectorStored = get_embeddings(pdf)
        st.write(f"{store_name}")

    query = st.text_area("Ask a question about the document", on_change=clear_submit)
    button = st.button("Submit")
    if button or st.session_state.get("submit"):
        st.session_state["submit"] = True
        # Output Columns
        answer_col, sources_col = st.columns(2)

        with get_openai_callback() as cb:
            sources = search_docs(VectorStored, query)
            try:
                answer = get_response_from_query(sources, query)
                print(cb)

                with answer_col:
                    st.markdown("#### Answer")
                    # st.write(answer)
                    st.markdown(answer["text"].split("Sources :")[0])

                with sources_col:
                    st.markdown("#### Sources")
                    st.markdown(answer["text"].split("Sources :")[-1])

            except OpenAIError as e:
                st.error(e._message)


if __name__ == "__main__":
    main()
