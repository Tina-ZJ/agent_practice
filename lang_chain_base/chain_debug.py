#!/usr/bin/env python
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain.globals import set_verbose,set_debug
from langsmith import traceable,Client
from dotenv import load_dotenv
load_dotenv(override=True)


# define llm
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

# define tools
tools = [TavilySearchResults(max_results=1)]

# define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# define agent
agent = create_tool_calling_agent(llm, tools, prompt)

def verbose():
    set_verbose(True)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    agent_executor.invoke(
        {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
    )

def debug():
    set_debug(True)
    set_verbose(False)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    agent_executor.invoke(
        {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
    )

def add_dataset():
    client = Client()
    dataset_name = "QA Example Dataset Lily"
    dataset = client.create_dataset(dataset_name)
    client.create_examples(
        inputs=[
            {"question": "What is LangChain?"},
            {"question": "What is LangSmith?"},
            {"question": "What is OpenAI?"},
            {"question": "What is Google?"},
            {"question": "What is Mistral?"},
        ],
        outputs=[
            {"answer": "A framework for building LLM applications"},
            {"answer": "A platform for observing and evaluating LLM applications"},
            {"answer": "A company that creates Large Language Models"},
            {"answer": "A technology company known for search"},
            {"answer": "A company that creates Large Language Models"},
        ],
        dataset_id=dataset.id,
    )

def search_agent():
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    result = agent_executor.invoke(
        {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
    )
    return result



if __name__=='__main__':
    evaluator = search_agent()
    print(evaluator)
    # verbose()
    # debug()