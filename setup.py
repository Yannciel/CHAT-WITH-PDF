from distutils.core import setup

setup(
    name="chat-with-pdf",
    version="1.0",
    description="Python pacakge that allows to chat with a pdf",
    author="Mingqiang Wang",
    author_email="mingqiang.wang.nlp@gmail.com",
    packages=["streamlit~=1.23.1", "langchain~=0.0.201"],
)
