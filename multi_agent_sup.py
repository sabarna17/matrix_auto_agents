import getpass
import os

from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_experimental.utilities import PythonREPL
repl = PythonREPL()
from langchain_core.tools import tool

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

os.environ['TAVILY_API_KEY'] = "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"
os.environ['GROQ_API_KEY'] = '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'

# _set_if_undefined("OPENAI_API_KEY")
# _set_if_undefined("TAVILY_API_KEY")

from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

from langchain_core.messages import HumanMessage


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }
    
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal

members = [
    "Researcher", 
    "quickchart", 
    # "UI Automation",
    # "CSV Creator"
]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members

NextType = Literal[tuple(options)]

class routeResponse(BaseModel):
    # next: Literal[*options]
    next: NextType


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))




# llm = ChatOpenAI(model="gpt-4o")
from langchain_groq import ChatGroq 
from groq import Groq
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
        azure_endpoint='https://azure-oai-swiftsnow.openai.azure.com/', #os.environ['AZURE_OPENAI_ENDPOINT_NEW'],
        openai_api_version='2024-08-01-preview', #os.environ['OPENAI_API_VERSION_NEW'],
        openai_api_key='69ee6927fadc423197bd4b88140c9ff9', #os.environ['AZURE_OPENAI_API_KEY_NEW'],
        deployment_name='azure-oai-swift-gpt-4', #os.environ['AZURE_DEPLOYMENT_NEW'], 
        max_tokens=600, max_retries=3,verbose=True)  

# llm = ChatGroq(
#         model="llama-3.1-70b-versatile",
#         temperature=1,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         stop_sequences='end'
#     )

def supervisor_agent(state):
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    return supervisor_chain.invoke(state)

import functools
import operator
from typing import Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


research_agent = create_react_agent(llm, tools=[tavily_tool])
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION


workflow = StateGraph(AgentState)

from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
# You can create the tool to pass to an agent
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="""
    A Python shell. 
    Use this to execute python commands to create any charts. 
    Use quickchart.io library and python code to execute to create graph and save locally. 
    Input should be a valid python script. 
    If you want to see the output of a value, you should print it out with `print(...)`.
    """
    ,
    func=python_repl.run,
)

chart_node = create_react_agent(llm, tools=[python_repl_tool])

# @tool
# def python_repl_chart(code: Annotated[str, "Use quickchart.io library and python code to execute to create graph and save locally"]):
#     """Use this to execute python code.
#     filepath = './' + filename
#     Use quickchart.io library and python code to execute to create graph and save locally
#     If you want to see the output of a value, you should execute the code and print it out with `print(...)`.
#     This is visible to the user."""
#     try:
#         print('Code::\n',code,'\n------------------')
#         result = repl.run(code)
#     except BaseException as e:
#         return f"Failed to execute. Error: {repr(e)}"
#     return f"Succesfully executed:\\\\n`python\\\\\\\\n{code}\\\\\\\\n`\\\\nStdout: {result}"
# tools = [python_repl_chart]
# tool_executor = ToolExecutor(tools)
# chart_creator= create_react_agent(
#     llm,
#     tool_executor
# )
# chart_node= functools.partial(agent_node, agent=chart_creator, name="quickchart")


# coder= create_react_agent(
#     llm,
#     [python_repl_tool]
# )
# chart_node= functools.partial(agent_node, agent=coder, name="Chart Image Generator")
# # # # # # # # # 


# # # # # # # # # 
@tool
def python_repl_ui(code: Annotated[str, "The python code only to execute to automate UI with selenium."]):
    """Use this to execute python code. The python code only to execute to automate UI with selenium.
    Use -> webdriver.Chrome() 
    If you want to see the output of a value, you should execute the code and print it out with `print(...)`.
    This is visible to the user."""
    try:
        print('Code:: \n', code, '\n------------------')
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\\\\n`python\\\\\\\\n{code}\\\\\\\\n`\\\\nStdout: {result}"
tools = [python_repl_ui]
tool_executor = ToolExecutor(tools)
ui_automator= create_react_agent(
    llm,
    tool_executor
)
selenium_node= functools.partial(agent_node, agent=ui_automator, name="UI Automation")
# # # # # # # # # 

# # # # # # # # # 
@tool
def python_repl_csv(code: Annotated[str, "The python code to execute to create CSV or excel files"]):
    """Use this to execute python code.
    filepath = './' + filename
    your task is to create an CSV using pandas library file with the proper input
    If you want to see the output of a value, you should execute the code and print it out with `print(...)`.
    This is visible to the user."""
    try:
        print('Code:: \n', code, '\n------------------')
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\\\\n`python\\\\\\\\n{code}\\\\\\\\n`\\\\nStdout: {result}"
tools = [python_repl_csv]
tool_executor = ToolExecutor(tools)
csv_creator= create_react_agent(
    llm,
    tool_executor
)
csv_node= functools.partial(agent_node, agent=csv_creator, name="CSV Creator")

from langgraph.pregel import RetryPolicy

workflow.add_node("Researcher", research_node)
workflow.add_node("quickchart", chart_node,retry=RetryPolicy(max_attempts=2))
# workflow.add_node("UI Automation", selenium_node,retry=RetryPolicy(max_attempts=2))
# workflow.add_node("CSV Creator", csv_node,retry=RetryPolicy(max_attempts=2))
workflow.add_node("supervisor", supervisor_agent)


for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
print('-----------------------------')
print(conditional_map)
print('-----------------------------')
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

for s in graph.stream(
    {
        "messages": [
            # HumanMessage(content="Code hello world and print it to the terminal")
            HumanMessage(                
                content="Fetch India GDP for last 6 years,"
                " use quickchart.io library to create a bar graph for the data and create a local image file with the image."
                "once completed. finish"
                # "Open chrome and goto gmail and send an email to sabarnapass@gmail attaching the image."
                # "Once email sent, finish." 
                
                # content = "Open chrome and then open youtube url."
                # "then search youtube channel 'Sarupyo Chatterjee' who plays guitar. "
                # "Fetch all the urls of the video published from that channel."
                # "Print all the urls in the console, Finish"
                
                # "Then create a local csv file to create a list with video title, url"
                # "then, finish" 
                # "then open youtube"
                # "then search youtube channel 'Sarupyo Chatterjee' "
                # "like the first video which is opened" 
            )
        ]
    },
    {"recursion_limit": 400}
):
    if "__end__" not in s:
        print(s)
        print("----")