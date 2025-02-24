import os
import yaml
from crewai import Agent, Task, Crew, Process
from typing import List, Dict, Tuple, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(..., description="The search query to execute")

class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web for current information on a topic. Input should be a specific search query."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchAPIWrapper()
        return search.run(query)

class DeepSearchInput(BaseModel):
    """Input schema for deep search tool."""
    query: str = Field(..., description="The search query to execute")

class DeepSearchTool(BaseTool):
    name: str = "Deep Search"
    description: str = "Perform a thorough web search returning multiple detailed results. Input should be a specific search query."
    args_schema: Type[BaseModel] = DeepSearchInput

    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.results(query, max_results=5)
        return "\n\n".join([f"Source: {r['link']}\nTitle: {r['title']}\nSnippet: {r['snippet']}" for r in results])

def load_config(file_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def create_agents() -> Tuple[List[Agent], Dict[str, Agent]]:
    """Create agents and return both a list and a name-to-agent mapping."""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'agents.yaml')
    config = load_config(config_path)
    
    agents = []
    agent_map = {}
    
    # Create tool instances
    tools_map = {
        "Web Search": WebSearchTool(),
        "Deep Search": DeepSearchTool()
    }
    
    for agent_config in config['agents']:
        # Create tools for this agent based on config
        tools = []
        if 'tools' in agent_config and agent_config['tools']:
            for tool_config in agent_config['tools']:
                tool_name = tool_config['name']
                if tool_name in tools_map:
                    tools.append(tools_map[tool_name])
        
        agent = Agent(
            role=agent_config['role'],
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            verbose=True,
            tools=tools
        )
        agents.append(agent)
        agent_map[agent_config['name']] = agent
    
    return agents, agent_map

def create_tasks(agent_map: Dict[str, Agent], research_question: str) -> List[Task]:
    """Create tasks using the agent mapping."""
    tasks = []
    
    # Create research planning task
    tasks.append(Task(
        description=f"Research Question: {research_question}\n\nAnalyze this research question and create a detailed plan. Break it down into specific areas to investigate and create a structured approach. Consider what types of sources would be most valuable.",
        agent=agent_map['coordinator'],
        expected_output="A structured research plan including: 1) Key areas to investigate, 2) Types of sources to prioritize, 3) Specific questions to answer"
    ))
    
    # Create source finding task
    tasks.append(Task(
        description=f"Find relevant and reliable sources about: {research_question}\nUse your web search tools to find authoritative sources that directly address the research areas.",
        agent=agent_map['search_agent'],
        expected_output="A curated list of sources with: 1) Full citations, 2) Brief description of relevance, 3) Initial assessment of reliability"
    ))
    
    # Create content extraction task
    tasks.append(Task(
        description="Use your web search tools to verify and extract key information from the identified sources. Focus on findings, methodologies, and conclusions.",
        agent=agent_map['content_extractor'],
        expected_output="Organized notes containing: 1) Key findings, 2) Relevant quotes or statistics, 3) Summary of main arguments"
    ))
    
    # Create analysis task
    tasks.append(Task(
        description=f"Using your web search tools to fact-check and validate, analyze the extracted information to answer: {research_question}\nIdentify patterns, evaluate evidence, and synthesize findings.",
        agent=agent_map['analyst'],
        expected_output="Analysis report highlighting: 1) Key findings and their significance, 2) Evidence quality, 3) Patterns and trends"
    ))
    
    return tasks

def create_crew(research_question: str = "What are the latest developments in quantum computing?") -> Crew:
    """Create the crew with the configured agents and tasks."""
    agents, agent_map = create_agents()
    tasks = create_tasks(agent_map, research_question)
    
    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
