from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field  # Use BaseModel from pydantic (v2 required by langchain)
from langchain.schema import Document
from langgraph.graph import StateGraph, END

from src.document_processor import DocumentProcessor
from src.data_extractor import DataExtractor
from src.vector_db_manager import VectorDBManager
from src.data_analysis_agent import DataAnalysisAgent

class MultiAgentWorkflow:
    """Coordinates the multi-agent workflow using LangGraph."""
    
    def __init__(self, openai_api_key: str, pinecone_api_key: str):
        self.document_processor = DocumentProcessor()
        self.data_extractor = DataExtractor()
        self.vector_db_manager = VectorDBManager(api_key=pinecone_api_key)
        self.data_analyzer = DataAnalysisAgent()
        
        # Initialize the LangGraph workflow
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph multi-agent workflow."""
        class WorkflowState(BaseModel):
            files: List[str] = Field(default_factory=list)
            documents: List[Document] = Field(default_factory=list)
            chunks: List[Document] = Field(default_factory=list)
            quantitative_data: Dict[str, Any] = Field(default_factory=dict)
            qualitative_data: Dict[str, str] = Field(default_factory=dict)
            analysis: Dict[str, Any] = Field(default_factory=dict)
            summary: str = ""
            current_status: str = "initialized"
            error: Optional[str] = None
        
        def process_documents(state: WorkflowState) -> WorkflowState:
            try:
                state.documents = self.document_processor.extract_from_multiple_files(state.files)
                state.current_status = "documents_processed"
            except Exception as e:
                state.error = f"Error processing documents: {str(e)}"
                state.current_status = "error"
            return state
        
        def chunk_documents(state: WorkflowState) -> WorkflowState:
            try:
                state.chunks = self.data_extractor.split_documents(state.documents)
                state.current_status = "documents_chunked"
            except Exception as e:
                state.error = f"Error chunking documents: {str(e)}"
                state.current_status = "error"
            return state
        
        def store_in_vector_db(state: WorkflowState) -> WorkflowState:
            try:
                self.vector_db_manager.store_documents(state.chunks)
                state.current_status = "documents_stored"
            except Exception as e:
                state.error = f"Error storing in vector DB: {str(e)}"
                state.current_status = "error"
            return state
        
        def extract_data(state: WorkflowState) -> WorkflowState:
            try:
                # Combine all document chunks for extraction
                all_text = "\n\n".join([chunk.page_content for chunk in state.chunks])
                state.quantitative_data = self.data_extractor.extract_quantitative_data(all_text)
                state.qualitative_data = self.data_extractor.extract_qualitative_data(all_text)
                state.current_status = "data_extracted"
            except Exception as e:
                state.error = f"Error extracting data: {str(e)}"
                state.current_status = "error"
            return state
        
        def analyze_data(state: WorkflowState) -> WorkflowState:
            try:
                state.analysis = self.data_analyzer.analyze_data(
                    state.quantitative_data, 
                    state.qualitative_data
                )
                state.current_status = "data_analyzed"
            except Exception as e:
                state.error = f"Error analyzing data: {str(e)}"
                state.current_status = "error"
            return state
        
        def generate_summary(state: WorkflowState) -> WorkflowState:
            try:
                state.summary = self.data_analyzer.generate_summary_report(state.analysis)
                state.current_status = "summary_generated"
            except Exception as e:
                state.error = f"Error generating summary: {str(e)}"
                state.current_status = "error"
            return state
        
        # Create the workflow graph using the StateGraph wrapper
        workflow = StateGraph(WorkflowState)
        workflow.add_node("process_documents", process_documents)
        workflow.add_node("chunk_documents", chunk_documents)
        workflow.add_node("store_in_vector_db", store_in_vector_db)
        workflow.add_node("extract_data", extract_data)
        workflow.add_node("analyze_data", analyze_data)
        workflow.add_node("generate_summary", generate_summary)
        
        workflow.add_edge("process_documents", "chunk_documents")
        workflow.add_edge("chunk_documents", "store_in_vector_db")
        workflow.add_edge("store_in_vector_db", "extract_data")
        workflow.add_edge("extract_data", "analyze_data")
        workflow.add_edge("analyze_data", "generate_summary")
        workflow.add_edge("generate_summary", END)
        
        workflow.set_entry_point("process_documents")
        self.graph = workflow.compile()
    
    def run(self, file_paths: List[str]) -> Dict[str, Any]:
        """Run the multi-agent workflow on the provided files."""
        initial_state = {"files": file_paths}
        final_state = self.graph.invoke(initial_state)

        # Instead of this:
        # if final_state.error:

        # Use:
        if final_state.get("error"):
            return {
                "status": "error",
                "error": final_state["error"],
                "current_stage": final_state.get("current_status")
            }
        else:
            return {
                "status": "success",
                "quantitative_data": final_state.get("quantitative_data"),
                "qualitative_data": final_state.get("qualitative_data"),
                "analysis": final_state.get("analysis"),
                "summary": final_state.get("summary")
            }

