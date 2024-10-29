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
    # "Coder", 
    "ChromeUIAutomation",
    # "UIActionValidator"
    "CSVCreator"
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
llm_code = AzureChatOpenAI(
        azure_endpoint='https://azure-oai-swiftsnow.openai.azure.com/', #os.environ['AZURE_OPENAI_ENDPOINT_NEW'],
        openai_api_version='2024-08-01-preview', #os.environ['OPENAI_API_VERSION_NEW'],
        openai_api_key='69ee6927fadc423197bd4b88140c9ff9', #os.environ['AZURE_OPENAI_API_KEY_NEW'],
        deployment_name='azure-oai-swiftsnow-gpt-4o', #os.environ['AZURE_DEPLOYMENT_NEW'], 
        # deployment_name='azure-oai-swift-gpt-4', #os.environ['AZURE_DEPLOYMENT_NEW'], 
        max_tokens=600, 
        max_retries=2,
        # verbose=True
    )  
llm = AzureChatOpenAI(
        azure_endpoint='https://azure-oai-swiftsnow.openai.azure.com/', #os.environ['AZURE_OPENAI_ENDPOINT_NEW'],
        openai_api_version='2024-08-01-preview', #os.environ['OPENAI_API_VERSION_NEW'],
        openai_api_key='69ee6927fadc423197bd4b88140c9ff9', #os.environ['AZURE_OPENAI_API_KEY_NEW'],
        deployment_name='azure-oai-swift-gpt-4', #os.environ['AZURE_DEPLOYMENT_NEW'],  
        max_tokens=600, 
        max_retries=2,
        # verbose=True
)  

llm_groq = ChatGroq(
        model="llama-3.2-90b-text-preview",
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        stop_sequences='end'
    )

def supervisor_agent(state):
    supervisor_chain = prompt | llm_groq.with_structured_output(routeResponse)
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


research_agent = create_react_agent(llm_groq, tools=[tavily_tool])
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
    Input should be a valid python script. 
    If you want to see the output of a value, you should print it out with `print(...)`.
    """    ,
    func=python_repl.run,
)
code_node = create_react_agent(llm_code, tools=[python_repl_tool])

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
def python_repl_ui(
        code: Annotated[str, "The python code only to execute to automate UI with selenium."], 
        functionality: Annotated[str, "The functionality of the python code to automate UI."] ):
    """Use this to execute python code. The python code only to execute to automate UI with selenium.
    Use the below code for chrome UI automation -> 
    
    `from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--start-fullscreen')
    driver = webdriver.Chrome(options=options)`
    
    for sap btp login, you can use user id - `sabarna17@gmail.com`and password - `Sabu@2024`
    
    If you want to see the output of a value, you should execute the code and print it out with `print(...)`.
    This is visible to the user."""
    try:
        print('Code:: \n', code, '\n------------------')
        print('functionality:: \n', functionality, '\n------------------')
        result = repl.run(code)
    # except BaseException as e:
    #     return f"Failed to execute. Error: {repr(e)}"
        
        feedback = screen_validator(functionality)
    
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"executed:\\\\n`python\\\\\\\\n{code}\\\\\\\\n`\\\\n Then received the feedback from Vision AI: {feedback} \n Stdout: {result}. \n`\\\\"

import pyautogui
import datetime
from read_ss_with_gemini import generate_sc_desc

def take_screenshot():
    # Get the current time to use as part of the file name
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the file name
    file_name = f"screenshot_{current_time}.png"
    # Take the screenshot
    screenshot = pyautogui.screenshot()
    # Save the screenshot
    screenshot.save(file_name)
    print(f"Screenshot saved as {file_name}")
    return file_name

# @tool
            # - application_name
            # - application_details: functionality of this application
            # - ui_filters: if user selected some filters for fetching the data
            # - fields: this is the list of fields inside the screen
            # - section_headers: sections in the application
            # - chart: if any chart visible then fill this details
            # - table: fill the details of the table with the description if visible
            # - ui_errors: if any errors are visible then fill this details             
            # - contents: videos, images
@tool
def screen_validator(expectation: Annotated[str, "The python code only to execute to analyze and validate UI screen with Vision AI."]):
                                            # This function takes the screenshot of the desktop and tries to analyze what screen it is displaying using Google Vision AI.
    """
    The python code only to execute to analyze and validate UI screen with Vision AI.
    Validate the response output for your further processing.
    """
    try:
        file_name = take_screenshot()
        prompt = f"""
            Your task is to analyze the Chrome browser from the image given and validate accurately and then find the deviation from the expectation below. 
            Provide very specific functionality instructions to rectify the automation code without feedback, comments. 
            Expectation: {expectation}
        
        """
        resp = generate_sc_desc(file_name,prompt)
        print(resp)
        return resp
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    
tools = [python_repl_ui] # screen_validator,

tool_executor = ToolExecutor(tools)
ui_automator= create_react_agent(
    llm,
    tool_executor
)
selenium_node= functools.partial(agent_node, agent=ui_automator, name="ChromeUIAutomation")
# # # # # # # # # 

# tools = [screen_validator] # screen_validator,

# tool_executor = ToolExecutor(tools)
# ui_automator= create_react_agent(
#     llm,
#     tool_executor
# )
# screen_validator_node= functools.partial(agent_node, agent=ui_automator, name="UIActionValidator")

# # # # # # # # 
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
        
        print('Result ::\n', result, '\n------------------')
        
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\\\\n`python\\\\\\\\n{code}\\\\\\\\n`\\\\nStdout: {result}"
tools = [python_repl_csv]
tool_executor = ToolExecutor(tools)
csv_creator= create_react_agent(
    llm,
    tool_executor
)
csv_node= functools.partial(agent_node, agent=csv_creator, name="CSVCreator")

from langgraph.pregel import RetryPolicy

workflow.add_node("Researcher", research_node,retry=RetryPolicy(max_attempts=2))
workflow.add_node("ChromeUIAutomation", selenium_node,retry=RetryPolicy(max_attempts=2))
# workflow.add_node("UIActionValidator", screen_validator_node,retry=RetryPolicy(max_attempts=2))
# workflow.add_node("Coder", code_node,retry=RetryPolicy(max_attempts=2))
workflow.add_node("CSVCreator", csv_node,retry=RetryPolicy(max_attempts=2))
workflow.add_node("supervisor", supervisor_agent)


for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
print('Agent Workflow-----------------------------')
print(conditional_map)
print('-----------------------------')
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()


def few_shot(requirement):
    case = ''
    
    prompt = f"""
    
    Follow the example and give the response without any feedback comment of the user_input given at the bottom.
    
    Examples": 
    1.  user_input: "hi"
        ai_response: "case~1"
        
    2.  user_input: "how are you"
        ai_response: "case~1"        
    
    3.  user_input: "play a song "
        ai_response: "case~1"        
        
    4.  user_input: "hi"
        ai_response: "case~1"        
       
    5.  user_input: "hi"
        ai_response: "case~1"        
        
    6.  user_input: "hi"
        ai_response: "case~1"        
        
    7.  user_input: "hi"
        ai_response: "case~1"                                            
    
    user_input: f{requirement}
    
    """
    
    llm_groq = ChatGroq(
        model="llama-3.2-90b-text-preview",
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        stop_sequences='end'
    )    
    resp = llm_groq.invoke(prompt)
 

    return case

def grok_task_creation(requirement):
    prompt = f"""
    
    If user asks for a perform a task then follow the below instructions:
            
            Your task is to convert the requirement to automated tasks using the tools like - members. 
            Your output should should be precised, small and to the point without feedback and extra comments.
            You can also refer to the example give at the bottom.
            
            user_input:
            f{requirement}
            
            
            Example 1:
                user_input - 
                        Play a most famous sitar in indian classical music 
                
                Response - 
                        1. Fetch Famous Sitar player who plays indian clasical song
                        2. Open Chrome and then open Youtube. 
                        3. Then play the entire song of that famous Sitar player
                        

            Example 2:
                user_input - 
                        Get latest news from west bengal and then search it in youtube to get video urls
                
                Response - 
                        1. You can research and get the trending news of west bengal
                        2. Open Chrome and then open Youtube and search for the news title. 
                        3. Pick up the top 3 news video url from your search and then save it to local csv file.
    
    If you see the task is unclear then ask follow up questions.
    
    """
    llm_groq = ChatGroq(
        model="llama-3.2-90b-text-preview",
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        stop_sequences='end'
    )    
    resp = llm_groq.invoke(prompt)
    # print(resp)
    return resp.content


def call_agent(steps):
    # steps = grok_task_creation(requirement)
    last = ''
    print(steps)
    for s in graph.stream({"messages":[HumanMessage(content = steps)]},{"recursion_limit":400}):
        if "__end__" not in s:
            print(s)
            last = s
            print("------------------------------------------------------------------------")
        else:
            break          
    return last


# steps = """
#     1. Fetch Famous Sitar player who plays indian clasical song
#     2. Open Chrome and then open Youtube. 
#     3. Then play the entire song of that famous Sitar player 
# """
# call_agent(steps)

# import streamlit as st
# if "scenario" not in st.session_state:
#     st.session_state["scenario"] = ''

# # Handle the main scenario text area
# st.session_state.scenario = st.text_area(
#     'scenario',
#     placeholder='Scenario',
#     value=st.session_state.scenario,
#     label_visibility='collapsed'
# )

# # Handle the "⚙️ Prepare Steps" button
# if st.button("📥 Start Agent"):
#     if st.session_state.scenario:
#         # st.session_state.scenario_steps = grok_task_creation(st.session_state.scenario)
#         # whatif_steps(st, st.session_state.scenario_steps)
#         call_agent(st.session_state.scenario)
#     else:
#         st.toast('Please enter a scenario')


#     {
#         "messages": [
#             # HumanMessage(content="Code hello world and print it to the terminal")
#             HumanMessage(                
#                 content = """
#                 1. Fetch Famous Sitar player who plays indian clasical song
#                 2. Open Chrome and then open Youtube. 
#                 3. Then play the entire song of that famous Sitar player 
#                 """
#                 # content = """
#                 # 1. Open chrome and open google search engine and then search for sap btp login. 
#                 # 2. Click on the SAP BTP login url from the search history.
#                 # 3. Use the credentials sabarna17@gmail.com , `Sabu@2024` to login to SAP BTP
#                 # 4. Then goto trial account 
#                 # """
                
#                 # content = """
#                 # 1. Open command prompt and then initiae login to sap btp cloud foundry with command cf8 login.
#                 # 2. Use the credentials sabarna17@gmail.com , `Sabu@2024` to login cf8
#                 # 3. then list out the sap btp apps with command cf8 apps
#                 # """
#                 # Play any indian classical song with contain Sitar played by the most famous Sitar player in youtube.'
                
#                 # content=
#                 # "1. Fetch India GDP for last 6 years,"
#                 # "2. use quickchart.io library to create a bar graph for the data and create a local image file with the name `india_gdp_growth_chart.png`."
                
#                 # "3. Open chrome and then open google.com url for image search."
#                 # "4. Then perform google search with the same file `india_gdp_growth_chart.png`."
                
#                 # "once completed. finish"
                
#                 # "Once email sent, finish." 
                
#                 # content = "Open chrome and then open youtube url."
#                 # "then the most popular song by jhonny Cash"
                
#                 # content = 'Play the most popular song of john mayor'
                
#                 # "Fetch all the urls of the video published from that channel."
#                 # "Print all the urls in the console, Finish"
                
#                 # "Then create a local csv file to create a list with video title, url"
#                 # "then, finish" 
#                 # "then open youtube"
#                 # "then search youtube channel 'Sarupyo Chatterjee' "
#                 # "like the first video which is opened" 
#             )
#         ]
#     },

    # return 


# for s in graph.stream(
#     {
#         "messages": [
#             # HumanMessage(content="Code hello world and print it to the terminal")
#             HumanMessage(                
#                 content = """
#                 1. Fetch Famous Sitar player who plays indian clasical song
#                 2. Open Chrome and then open Youtube. 
#                 3. Then play the entire song of that famous Sitar player 
#                 """
#                 # content = """
#                 # 1. Open chrome and open google search engine and then search for sap btp login. 
#                 # 2. Click on the SAP BTP login url from the search history.
#                 # 3. Use the credentials sabarna17@gmail.com , `Sabu@2024` to login to SAP BTP
#                 # 4. Then goto trial account 
#                 # """
                
#                 # content = """
#                 # 1. Open command prompt and then initiae login to sap btp cloud foundry with command cf8 login.
#                 # 2. Use the credentials sabarna17@gmail.com , `Sabu@2024` to login cf8
#                 # 3. then list out the sap btp apps with command cf8 apps
#                 # """
#                 # Play any indian classical song with contain Sitar played by the most famous Sitar player in youtube.'
                
#                 # content=
#                 # "1. Fetch India GDP for last 6 years,"
#                 # "2. use quickchart.io library to create a bar graph for the data and create a local image file with the name `india_gdp_growth_chart.png`."
                
#                 # "3. Open chrome and then open google.com url for image search."
#                 # "4. Then perform google search with the same file `india_gdp_growth_chart.png`."
                
#                 # "once completed. finish"
                
#                 # "Once email sent, finish." 
                
#                 # content = "Open chrome and then open youtube url."
#                 # "then the most popular song by jhonny Cash"
                
#                 # content = 'Play the most popular song of john mayor'
                
#                 # "Fetch all the urls of the video published from that channel."
#                 # "Print all the urls in the console, Finish"
                
#                 # "Then create a local csv file to create a list with video title, url"
#                 # "then, finish" 
#                 # "then open youtube"
#                 # "then search youtube channel 'Sarupyo Chatterjee' "
#                 # "like the first video which is opened" 
#             )
#         ]
#     },
#     {"recursion_limit": 400}
# ):
#     if "__end__" not in s:
#         print(s)
#         print("------------------------------------------------------------------------")
#     else:
#         break