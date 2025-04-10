import json
from typing import Dict, Any

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

class DataAnalysisAgent:
    """Agent responsible for analyzing and interpreting the extracted data."""
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    
    def analyze_data(self, quantitative_data: Dict[str, Any], qualitative_data: Dict[str, str]) -> Dict[str, Any]:
        """Analyze and interpret the combination of quantitative and qualitative data."""
        # Convert data to strings for the prompt
        quant_str = json.dumps(quantitative_data, indent=2)
        qual_str = json.dumps(qualitative_data, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a data analysis expert. Analyze the provided quantitative and qualitative data to generate insights. Look for patterns, relationships, and key findings. Provide a comprehensive analysis in a structured JSON format with sections for 'key_findings', 'trends', 'implications', and 'recommendations'."),
            HumanMessage(content=f"Analyze the following data:\n\nQuantitative Data:\n{quant_str}\n\nQualitative Data:\n{qual_str}")
        ])
        
        # response = self.llm.invoke(prompt)
        formatted = prompt.format_messages()  # ✅ creates list[BaseMessage]
        response = self.llm.invoke(formatted)

        try:
            # Parse the analysis as JSON
            analysis = json.loads(response.content)
            return analysis
        except json.JSONDecodeError:
            # Return raw text if not valid JSON
            return {"raw_analysis": response.content}
    
    def generate_summary_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable summary report based on the analysis."""
        analysis_str = json.dumps(analysis, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a business intelligence specialist. Create a clear, concise executive summary report based on the provided data analysis. The report should highlight key findings, insights, trends, and actionable recommendations."),
            HumanMessage(content=f"Generate a summary report based on this analysis:\n\n{analysis_str}")
        ])
        
        # response = self.llm.invoke(prompt)
        formatted = prompt.format_messages()  # ✅ creates list[BaseMessage]
        response = self.llm.invoke(formatted)

        return response.content