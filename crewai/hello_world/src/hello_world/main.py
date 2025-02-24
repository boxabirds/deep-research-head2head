#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crewai import Crew
from .crew import create_crew

def main(research_question: str = "What are the latest developments in quantum computing?"):
    """
    Execute the deep research workflow with the specified research question.
    
    Args:
        research_question (str): The research question to investigate
    
    Returns:
        str: The final research report
    """
    # Create and execute the crew
    crew = create_crew(research_question)
    result = crew.kickoff()
    
    return result

if __name__ == "__main__":
    # Get research question from command line if provided
    research_question = sys.argv[1] if len(sys.argv) > 1 else "What are the latest developments in quantum computing?"
    result = main(research_question)
    print("\nFinal Research Report:")
    print("=" * 80)
    print(result)
