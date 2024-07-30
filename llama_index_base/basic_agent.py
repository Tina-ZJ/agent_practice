from llama_index.llms.anthropic import Anthropic
from llama_index.llms.mistralai import MistralAI
from llama_index.core.tools import FunctionTool,QueryEngineTool, ToolMetadata
from llama_index.core import PromptTemplate,SimpleDirectoryReader, VectorStoreIndex
from prompts.system_prompt import customize_prompt
from llama_index.embeddings.mistralai import MistralAIEmbedding
import os
from llama_index.core.agent import FunctionCallingAgentWorker,AgentRunner, ReActAgentWorker, ReActAgent



anthropic_api_key='****'
mistralai_api_key='fZxU1KNKA87Q3NAGgpUISFpRWaRy7HYP'

def multiply(a:int, b:int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

def add(a:int, b:int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

# tools
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
tools = [multiply_tool,add_tool]

# llm
# llm = Anthropic(model='claude-3-opus-20240229',api_key=anthropic_api_key)
llm = MistralAI(model="mistral-large-latest", api_key=mistralai_api_key)

def caculate_agent():
    agent_worker = FunctionCallingAgentWorker.from_tools(
        [multiply_tool, add_tool],
        llm = llm,
        verbose=True,
        allow_parallel_tool_calls=False,
    )
    agent = AgentRunner(agent_worker)
    question = "What is (121 * 3) + 42?"
    response = agent.chat(question)
    print(str(response))


def react_agent():
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
    # react_step_engine = ReActAgentWorker.from_tools(tools, llm=llm, verbose=True)
    # agent = AgentRunner(react_step_engine)
    # show system prompt
    prompt_dict = agent.get_prompts()
    # for k,v in prompt_dict.items():
    #     print(f"Prompt: {k}\n\nValue: {v.template}")
    question = "What is (121 * 3) + 42?"
    print("**********原始system prompt的结果*******************")
    agent.chat(question)

    # customizing the system prompt
    react_system_prompt = customize_prompt()
    agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})
    agent.reset()
    agent.get_prompts()
    prompt_dict = agent.get_prompts()
    # for k,v in prompt_dict.items():
    #     print(f"Prompt: {k}\n\nValue: {v.template}")
    question = "What is (121 * 3) + 42?"
    print("**********如下是修改system prompt后的结果*******************")
    agent.chat(question)


def rag_agent():
    # 获取当前执行文件的绝对路径
    file_path = os.path.abspath(__file__)
    directory = os.path.dirname(file_path)
    source_path = directory+'/data/10k/uber_2021.pdf'
    embed_model = MistralAIEmbedding(api_key=mistralai_api_key)
    query_llm = MistralAI(model="mistral-medium",api_key=mistralai_api_key)
    # load data
    uber_docs = SimpleDirectoryReader(
        input_files=[source_path]
    ).load_data()
    # build index
    uber_index = VectorStoreIndex.from_documents(
        uber_docs, embed_model=embed_model
    )
    uber_engine = uber_index.as_query_engine(similary_top_k=3,llm=query_llm)
    query_engine_tool = QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021."
                "Use a detailed plain text question as input to the tool"
            ),
        ),
    )
    agent_worker = FunctionCallingAgentWorker.from_tools(
    [query_engine_tool], llm=llm, verbose=True
    )
    agent = agent_worker.as_agent()
    response = agent.chat("Tell me both the risk factors and tailwinds for Uber? Do two parallel tool calls")
    print(str(response))


if __name__=='__main__':
    # caculate_agent()
    # react_agent()
    rag_agent()