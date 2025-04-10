from setuptools import setup, find_packages

setup(
    name="multi-agent-data-system",
    version="0.1.0",
    description="A multi-agent system for data extraction and analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langgraph>=0.0.15",
        "pinecone-client>=2.2.0",
        "langchain-pinecone>=0.0.1",
        "openai>=1.1.0",
        "pypdf2>=3.0.0",
        "PyPDF2>=3.0.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "openpyxl>=3.1.0",
        "xlrd>=2.0.1",
        "langchain-openai>=0.0.2",
        "streamlit>=1.24.0",
        "plotly>=5.13.0",
    ],
    python_requires=">=3.8",
)