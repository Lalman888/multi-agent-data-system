import os
import json
import dotenv

from src.multi_agent_workflow import MultiAgentWorkflow
from src.query_system import QuerySystem

def main():
    # Load environment variables
    dotenv.load_dotenv()
    
    # Get API keys from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        print("Error: Missing API keys. Please set OPENAI_API_KEY and PINECONE_API_KEY in your .env file.")
        return
    
    # Initialize the workflow
    workflow = MultiAgentWorkflow(
        openai_api_key=OPENAI_API_KEY,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    # List of files to process
    files = [
        "source/document1.pdf",
        "source/spreadsheet.xlsx",
        "source/contract.pdf"
    ]
    
    # Run the workflow
    results = workflow.run(files)
    
    # Print the results
    if results["status"] == "success":
        print("\n=== QUANTITATIVE DATA ===")
        print(json.dumps(results["quantitative_data"], indent=2))
        
        print("\n=== QUALITATIVE DATA ===")
        print(json.dumps(results["qualitative_data"], indent=2))
        
        print("\n=== ANALYSIS ===")
        print(json.dumps(results["analysis"], indent=2))
        
        print("\n=== SUMMARY REPORT ===")
        print(results["summary"])
        
        # Create a query system
        query_system = QuerySystem(workflow.vector_db_manager)
        agent = query_system.create_interactive_agent()
        
        # Interactive questioning
        while True:
            question = input("\nAsk a question about the data (or 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            
            answer = agent.invoke({"input": question})
            print(f"\nAnswer: {answer['output']}")
    else:
        print(f"Error: {results['error']}")
        print(f"Failed at stage: {results['current_stage']}")


if __name__ == "__main__":
    main()