from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI  # Use the updated ChatOpenAI import
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from src.vector_db_manager import VectorDBManager

class QuerySystem:
    """Provides an interface for querying the processed data."""
    
    def __init__(self, vector_db_manager: VectorDBManager, llm=None):
        self.vector_db_manager = vector_db_manager
        self.llm = llm or ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
        self.retriever = vector_db_manager.create_retriever()
        
        # Create a QA chain for answering questions
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever
        )
    
    def ask(self, question: str) -> str:
        """Ask a question about the processed data."""
        return self.qa_chain.run(question)
    
    def create_interactive_agent(self) -> AgentExecutor:
        """Create an interactive agent for querying data."""
        # Define tools for the agent with a valid name (no spaces)
        tools = [
            Tool(
                name="SearchDocuments",  # Changed from "Search Documents" to "SearchDocuments"
                func=self.ask,
                description="Search for information in the processed documents. Useful for answering questions about document content."
            )
        ]
        
        # Create the agent prompt ensuring that it includes agent_scratchpad
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful data assistant that uses provided tools to answer questions and cites your sources."),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")  # Ensures the required agent_scratchpad variable is present
        ])
        
        # Create the agent
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        return agent_executor
