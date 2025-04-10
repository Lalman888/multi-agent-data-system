import re
import json
from typing import Dict, List, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage

class DataExtractor:
    """Agent responsible for extracting structured data from documents."""
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into manageable chunks."""
        return self.text_splitter.split_documents(documents)
    
    def _clean_json_response(self, text: str) -> str:
        """
        Remove markdown code block delimiters (like ```json ... ```) and extra whitespace.
        """
        # This regex captures text inside triple backticks optionally preceded by "json"
        pattern = r"^```(?:json)?\s*([\s\S]+?)\s*```$"
        match = re.search(pattern, text.strip(), re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    def extract_quantitative_data(self, text: str) -> Dict[str, Any]:
        """Extract numerical data and statistics from text."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a data extraction specialist. Extract all numerical data and statistics from the given text. Format the output as a JSON object with descriptive keys and numerical values. Only include clearly defined numerical values."),
            HumanMessage(content=f"Extract quantitative data from the following text:\n\n{text}")
        ])
        
        formatted = prompt.format_messages()  # creates list[BaseMessage]
        response = self.llm.invoke(formatted)
        
        # Clean the response to remove markdown formatting if present
        cleaned_response = self._clean_json_response(response.content)
        
        try:
            # Try to parse the cleaned response as JSON
            extracted_data = json.loads(cleaned_response)
            return extracted_data
        except json.JSONDecodeError:
            print("Could not parse LLM response as JSON, using regex fallback")
            # If parsing fails, use regex fallback
            matches = re.findall(r'(\w+(?:\s+\w+){0,5}?):\s*(\d+(?:\.\d+)?)', text)
            return {key.strip(): float(value) for key, value in matches}
    
    def extract_qualitative_data(self, text: str) -> Dict[str, str]:
        """Extract key insights, findings, and qualitative information."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a qualitative data analyst. Extract key insights, findings, themes, and qualitative information from the given text. Format the output as a JSON object with categories as keys and descriptions as values."),
            HumanMessage(content=f"Extract qualitative insights from the following text:\n\n{text}")
        ])
        
        formatted = prompt.format_messages()  # creates list[BaseMessage]
        response = self.llm.invoke(formatted)
        
        # Clean the response to remove markdown/code block markers
        cleaned_response = self._clean_json_response(response.content)
        
        try:
            extracted_data = json.loads(cleaned_response)
            return extracted_data
        except json.JSONDecodeError:
            print("Could not parse LLM response as JSON, returning raw text")
            return {"raw_insights": cleaned_response}
