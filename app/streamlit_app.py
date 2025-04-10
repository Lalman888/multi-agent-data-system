import os
import json
import tempfile
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Add parent directory to path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multi_agent_workflow import MultiAgentWorkflow
from src.query_system import QuerySystem

# --- Streamlit page configuration ---
st.set_page_config(
    page_title="Multi-Agent Data Extraction and Analysis",
    page_icon="📊",
    layout="wide",
)

# --- Initialize session state ---
if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "query_system" not in st.session_state:
    st.session_state.query_system = None
if "results" not in st.session_state:
    st.session_state.results = None
if "files_processed" not in st.session_state:
    st.session_state.files_processed = False

# --- Function to initialize API keys ---
def init_api_keys():
    openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_key", "")
    pinecone_key = os.getenv("PINECONE_API_KEY") or st.session_state.get("pinecone_key", "")
    return openai_key, pinecone_key

# --- Function to process uploaded files ---
def process_files(uploaded_files, openai_key, pinecone_key):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_file_paths.append(file_path)
        
        # Initialize the multi-agent workflow
        workflow = MultiAgentWorkflow(
            openai_api_key=openai_key,
            pinecone_api_key=pinecone_key
        )
        
        with st.spinner("Processing files... This may take a few minutes."):
            results = workflow.run(temp_file_paths)
        
        if results["status"] == "success":
            st.session_state.workflow = workflow
            st.session_state.results = results
            st.session_state.files_processed = True
            
            # Create query system using the workflow's vector database manager
            st.session_state.query_system = QuerySystem(workflow.vector_db_manager)
            st.success("Files processed successfully!")
            return True
        else:
            st.error(f"Error: {results.get('error', 'Unknown error')}")
            st.error(f"Failed at stage: {results.get('current_stage', 'Unknown stage')}")
            return False

# --- Sidebar for API keys ---
with st.sidebar:
    st.title("API Configuration")
    openai_key, pinecone_key = init_api_keys()
    
    with st.form("api_keys_form"):
        st.session_state.openai_key = st.text_input(
            "OpenAI API Key", 
            value=openai_key,
            type="password",
            help="Enter your OpenAI API key"
        )
        st.session_state.pinecone_key = st.text_input(
            "Pinecone API Key", 
            value=pinecone_key,
            type="password",
            help="Enter your Pinecone API key"
        )
        submitted = st.form_submit_button("Save API Keys")
        if submitted:
            st.success("API Keys saved!")

# --- Main application UI ---
st.title("📊 Multi-Agent Data Extraction and Analysis")
st.markdown(
    """
    This application uses a multi-agent system to extract, analyze, and query data from various document formats.
    Upload your files below to get started.
    """
)

# File upload widget
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, Excel, CSV, Text)", 
    accept_multiple_files=True,
    type=["pdf", "xlsx", "xls", "csv", "txt"]
)

# Process files button
if uploaded_files and st.button("Process Files"):
    if not st.session_state.openai_key or not st.session_state.pinecone_key:
        st.warning("Please enter both API keys in the sidebar.")
    else:
        process_files(uploaded_files, st.session_state.openai_key, st.session_state.pinecone_key)

# --- Display results if files processed ---
if st.session_state.files_processed and st.session_state.results:
    results = st.session_state.results
    st.header("Analysis Results")
    
    quant_tab, qual_tab, analysis_tab, summary_tab, query_tab = st.tabs([
        "Quantitative Data", 
        "Qualitative Data", 
        "Analysis", 
        "Summary Report",
        "Query System"
    ])
    
    with quant_tab:
        st.subheader("Extracted Quantitative Data")
        st.json(results["quantitative_data"])
        try:
            quant_df = pd.DataFrame.from_dict(results["quantitative_data"], orient='index')
            quant_df.index.name = 'Metric'
            quant_df.reset_index(inplace=True)
            st.dataframe(quant_df)
            st.bar_chart(quant_df.set_index('Metric'))
        except Exception as e:
            st.info(f"Could not convert quantitative data to a chart: {e}")
    
    with qual_tab:
        st.subheader("Extracted Qualitative Data")
        st.json(results["qualitative_data"])
    
    with analysis_tab:
        st.subheader("Data Analysis")
        st.json(results["analysis"])
    
    with summary_tab:
        st.subheader("Summary Report")
        st.markdown(results["summary"])
    
    with query_tab:
        st.subheader("Ask Questions About Your Data")
        st.write("Use the query system below to ask questions about the processed documents.")
        query = st.text_input("Enter your question:")
        if query and st.button("Ask Question"):
            with st.spinner("Generating answer..."):
                agent = st.session_state.query_system.create_interactive_agent()
                answer = agent.invoke({"input": query})
                # Display answer output in markdown formatting
                st.markdown("### Answer")
                st.markdown(answer.get("output", "No answer returned."))

