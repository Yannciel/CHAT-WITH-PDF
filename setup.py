from setuptools import find_packages, setup

setup(
    name="chat-with-pdf",
    version="1.0.1",
    description="Python pacakge that allows to chat with a pdf",
    author="Mingqiang Wang",
    author_email="mingqiang.wang.nlp@gmail.com",
    packages= find_packages(),
    python_requries=">=3.8",
    install_requires=["streamlit~=1.23.1", "langchain~=0.0.201","PyPDF2~=3.0.1",
"python-dotenv~=1.0.0"],
)
