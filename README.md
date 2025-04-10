# Multi-Agent Data Extraction and Consolidation System

A comprehensive framework that uses LangChain, LangGraph, Pinecone, and PyPDF2 (along with other packages) to:

1. Extract data from various document formats (PDFs, Excel spreadsheets, contracts, CSV, text).
2. Process and transform the extracted data.
3. Store document embeddings in a Pinecone vector database.
4. Consolidate and present relevant qualitative and quantitative information.
5. Offer an interactive querying interface via LangChain agents and a Streamlit web UI.

## Features

- **Multi-format Document Processing:** Handle PDF, Excel, CSV, and text files.
- **Data Extraction:** Intelligent extraction of quantitative (numerical) and qualitative (descriptive) data.
- **Vector Database Integration:** Store and query embeddings using Pinecone.
- **Multi-Agent Orchestration:** Utilize LangGraph for coordinating the overall workflow.
- **Interactive Query System:** Ask questions about your data using LangChain agents.
- **Streamlit Interface:** Easy-to-use web interface for end-users.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Lalman888/multi-agent-data-system.git
   cd multi-agent-data-system
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**

   Install the project in editable mode (ensure your `setup.py` is configured correctly):

   ```bash
   pip install -e .
   ```

   Alternatively, if you maintain a `requirements.txt` file, generate it with:

   ```bash
   pip freeze > requirements.txt
   ```
   
   Make sure to have the following key packages (versions are managed by your dependencies):
   
   - `langchain` (>=0.3.x)
   - `langchain-openai`
   - `langgraph`
   - `pinecone` (the official package, not pinecone-client)
   - `pypdf2` or `pypdf`
   - `streamlit`
   - and others as specified in your project.

4. **Set Up API Keys:**

   - Copy `config/.env.example` to `.env` in the project root.
   - Add your OpenAI and Pinecone API keys to the `.env` file.

## Pinecone Setup

1. **Install the Official Pinecone Package:**

   ```bash
   pip install pinecone
   ```

2. **Initialize Pinecone in Your Code:**

   In your code (e.g., `vector_db_manager.py`), initialize Pinecone as follows:

   ```python
   from pinecone import Pinecone, ServerlessSpec

   # Instantiate the Pinecone client with your API key
   pc = Pinecone(api_key="your-api-key")

   # Create a Pinecone index if it doesn't exist
   index_name = "multi-agent-data"
   spec = ServerlessSpec(cloud="aws", region="us-east-1")
   existing_indexes = pc.list_indexes().names()
   if index_name not in existing_indexes:
       pc.create_index(
           name=index_name,
           dimension=1536,  # e.g., OpenAI embedding dimension
           metric="cosine",
           spec=spec
       )
   ```

## Usage

### Command Line Interface

Run the main script to process documents and interact with the system:

```bash
python main.py
```

### Streamlit Web Interface

1. **Run the App:**

   From the project root directory, launch the Streamlit app:

   ```bash
   streamlit run app/streamlit_app.py
   ```

2. **Using the App:**

   - Configure API keys via the sidebar.
   - Upload your documents (PDF, Excel, CSV, text).
   - Process files to perform extraction, analysis, and storage.
   - View results in separate tabs (Quantitative Data, Qualitative Data, Analysis, Summary Report).
   - Use the interactive Query System to ask questions about the processed data.

## Project Structure

```
multi-agent-data-system/
│
├── app/                          # Streamlit application
│   ├── streamlit_app.py          # Main Streamlit app
│   └── utils/                    # Optional utility functions for Streamlit
│       ├── __init__.py
│       └── st_utils.py
│
├── config/                       # Configuration files
│   └── .env.example              # Example environment file
│
├── src/                          # Source code for the project
│   ├── __init__.py
│   ├── document_processor.py     # Document loading & processing
│   ├── data_extractor.py         # Data extraction logic (quantitative & qualitative)
│   ├── vector_db_manager.py      # Pinecone vector database management
│   ├── data_analysis_agent.py    # Data analysis and summary generation
│   ├── multi_agent_workflow.py   # LangGraph workflow orchestration
│   └── query_system.py           # Interactive query interface
│
├── main.py                       # CLI entry point for the project
├── requirements.txt              # List of package dependencies
├── setup.py                      # Package setup script for installation
└── README.md                     # Project documentation
```

## Running Tests

(If applicable, add instructions on how to run tests, e.g., using pytest)

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

---

### How to Run the Streamlit App

1. **Ensure your Virtual Environment is Active:**  
   Activate your virtual environment (if not already active):
   
   ```bash
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Launch the App:**  
   Run the following command from the project root:

   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Interact with the Interface:**  
   The default browser will open a new tab with the Streamlit interface. Use the sidebar to enter your API keys, upload your files, and then use the main page to process and analyze data.

