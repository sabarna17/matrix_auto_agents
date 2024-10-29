import json
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union
from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq # from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict


from langchain_core.tools import tool
# langgraph.prebuilt.ToolNode

from langchain_community.tools.tavily_search import TavilySearchResults
import os
os.environ['TAVILY_API_KEY'] = "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"
tavily_tool = TavilySearchResults(max_results=5)

from langchain_groq import ChatGroq
from langchain_experimental.utilities import PythonREPL
from typing import Annotated



repl = PythonREPL()

os.environ['GROQ_API_KEY'] = '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'
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
        azure_endpoint='https://azure-oai-swiftsnow.openai.azure.com/', #os.environ['AZURE_OPENAI_ENDPOINT_NEW'],
        openai_api_version='2024-08-01-preview', #os.environ['OPENAI_API_VERSION_NEW'],
        openai_api_key='69ee6927fadc423197bd4b88140c9ff9', #os.environ['AZURE_OPENAI_API_KEY_NEW'],
        deployment_name='azure-oai-swift-gpt-4', #os.environ['AZURE_DEPLOYMENT_NEW'], 
        max_tokens=200, max_retries=3)  

# llm = ChatGroq(
#         model="llama-3.1-70b-versatile",
#         temperature=1,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         stop_sequences='END'
#     )


@tool
def python_repl(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Use this to execute python code. If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user."""
    try:
        print('Code:: \n', code, '\n------------------')
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\\\\n`python\\\\\\\\n{code}\\\\\\\\n`\\\\nStdout: {result}"

tools = [tavily_tool, python_repl]

# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    
tool_executor = ToolExecutor(tools)

def tool_node(state):
    """This runs tools in the graph
    It takes in an agent action and calls that tool and returns the result."""
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    # We can pass single-arg inputs by value
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


# Either agent can decide to end
def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        # The previus agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "end"
    return "continue"

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(api_key="<Your API Key here>")

# import json
# from langchain_core.messages import (
#     AIMessage,
#     BaseMessage,
#     ChatMessage,
#     FunctionMessage,
#     HumanMessage,
# )
from langchain_core.utils.function_calling import convert_to_openai_function
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langgraph.graph import END, StateGraph
# from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    functions = [convert_to_openai_function(t) for t in tools]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\\\\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

research_agent= create_agent(
    llm,
    [tavily_tool],
    system_message="You should provide accurate data for the chart generator to use.",
)

chart_agent= create_agent(
    llm,
    [python_repl],
    system_message="Any charts you display will be visible by the user.",
)

import functools

research_node= functools.partial(agent_node, agent=research_agent, name="Researcher")
chart_node= functools.partial(agent_node, agent=chart_agent, name="Chart Generator")

workflow= StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("Chart Generator", chart_node)
workflow.add_node("call_tool", tool_node)
workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "Chart Generator", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "Chart Generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "call_tool",
# Each agent node updates the 'sender' field# the tool calling node does not, meaning
# this edge will route back to the original agent# who invoked the tool
		lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "Chart Generator": "Chart Generator",
    },
)
workflow.set_entry_point("Researcher")
graph= workflow.compile()

for s in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Fetch the top 10 youtuber who has most subscribers,"
                " then draw a bar graph of it."
                " Once you code it up, finish."
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
):
    print(s)
    print("----")
