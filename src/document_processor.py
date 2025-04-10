import os
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.schema import Document

class DocumentProcessor:
    """Handles document loading and text extraction from various file formats."""
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load and extract text from PDF documents."""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_excel(file_path: str) -> List[Document]:
        """Load and extract data from Excel files."""
        loader = UnstructuredExcelLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_csv(file_path: str) -> List[Document]:
        """Load and extract data from CSV files."""
        loader = CSVLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """Load and extract data from text files."""
        loader = TextLoader(file_path)
        return loader.load()
    
    @staticmethod
    def get_loader_for_file(file_path: str):
        """Get appropriate loader based on file extension."""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return DocumentProcessor.load_pdf
        elif file_extension in ['.xlsx', '.xls']:
            return DocumentProcessor.load_excel
        elif file_extension == '.csv':
            return DocumentProcessor.load_csv
        elif file_extension == '.txt':
            return DocumentProcessor.load_text
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    @staticmethod
    def extract_from_multiple_files(file_paths: List[str]) -> List[Document]:
        """Process multiple files and return combined documents."""
        all_documents = []
        
        for file_path in file_paths:
            try:
                loader_func = DocumentProcessor.get_loader_for_file(file_path)
                documents = loader_func(file_path)
                
                # Add source metadata
                for doc in documents:
                    doc.metadata["source"] = file_path
                    doc.metadata["file_type"] = Path(file_path).suffix.lower()
                
                all_documents.extend(documents)
                print(f"Successfully processed: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        return all_documents