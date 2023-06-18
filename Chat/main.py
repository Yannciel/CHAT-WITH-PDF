import streamlit as st
import pickle

from openai.error import OpenAIError
from langchain.callbacks import get_openai_callback
from Models import pdf_reader, qa_llm
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
        text = pdf_reader.pdf_loader(pdf)
        VectorStored = pdf_reader.create_db_from_pdf_text(
            text
        )
        with open("{}.pkl".format(store_name_embeddings), "wb") as f:
            pickle.dump(VectorStored, f)
    return store_name,VectorStored


def main():
    st.header("Chat with PDF 💬")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf",on_change=clear_submit,
)

    # st.write(pdf)
    if pdf is not None:
        # # embeddings
        store_name, VectorStored = get_embeddings(pdf)
        st.write(f"{store_name}")
    else:
        raise ValueError("File type not supported!")
    
    query = st.text_area("Ask a question about the document", on_change=clear_submit)
    button = st.button("Submit")
    if button or st.session_state.get("submit"):
        st.session_state["submit"] = True
        # Output Columns
        answer_col, sources_col = st.columns(2)
        # sources = qa_llm.get_response_from_query(VectorStored, query)
        with get_openai_callback() as cb:
            sources = qa_llm.search_docs(VectorStored, query)
            print(cb)
        try:
            answer = qa_llm.get_answer(sources, query)

            with answer_col:
                st.markdown("#### Answer")
                st.markdown(answer["output_text"].split("SOURCES: ")[0])

            with sources_col:
                st.markdown("#### Sources")
                for source in sources:
                    st.markdown(source.page_content)
                    st.markdown(source.metadata["source"])
                    st.markdown("---")

        except OpenAIError as e:
            st.error(e._message)

if __name__ == '__main__':
    main()