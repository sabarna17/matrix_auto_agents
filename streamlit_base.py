import streamlit as st
import openai
from langchain_groq import ChatGroq 
from groq import Groq
import os
import logging
import extra_streamlit_components as stx
os.environ['GROQ_API_KEY'] = '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'


logger = logging.getLogger()
logging.basicConfig(encoding="UTF-8", level=logging.INFO)

st.set_page_config(page_title="What-IF Agent", page_icon="💡")

cookie_manager = stx.CookieManager(key="cookie_manager")

if "conversation_title" not in st.session_state:
    st.session_state["conversation_title"] = ''

# st.title("What-IF Agent")

scenario = ''
scenario_steps = ''
from multi_agent_v2 import grok_task_creation, call_agent


# Initialize session state variables if not present
if "conversation_title" not in st.session_state:
    st.session_state["conversation_title"] = ''

if "scenario" not in st.session_state:
    st.session_state["scenario"] = ''

if "scenario_steps" not in st.session_state:
    st.session_state["scenario_steps"] = ''

st.title("What-IF Agent")
tab1, tab2 = st.tabs(["Instruction Generator", "Agent Executor"])

with tab1:
    st.session_state.scenario = st.text_area(
        'scenario',
        placeholder='Scenario',
        value=st.session_state.scenario,
        label_visibility='collapsed'
    )

    # Handle the "⚙️ Prepare Steps" button
    if st.button("⚙️ Prepare Steps"):
        if st.session_state.scenario:
            st.session_state.scenario_steps = grok_task_creation(st.session_state.scenario)
            # whatif_steps(st, st.session_state.scenario_steps)
            # st.session_state.scenario_steps = st.text_area(
            #         'steps',
            #         placeholder='Steps',
            #         value=st.session_state.scenario_steps,
            #         label_visibility='collapsed',
            #         height=200
            #     )
        else:
            st.toast('Please enter a scenario')    

with tab2:
    st.session_state.scenario_steps = st.text_area(
                    'steps',
                    placeholder='Steps',
                    value=st.session_state.scenario_steps,
                    label_visibility='collapsed',
                    height=200
                )

    # Handle the "⚙️ Prepare Steps" button
    if st.button("⚙️ Prepare Steps"):
        if st.session_state.scenario_steps:
            response = call_agent(st.session_state.scenario_steps)
            # st.session_state.scenario_steps = grok_task_creation(st.session_state.scenario)
            # whatif_steps(st, st.session_state.scenario_steps)
            # st.session_state.scenario_steps = st.text_area(
            #         'steps',
            #         placeholder='Steps',
            #         value=st.session_state.scenario_steps,
            #         label_visibility='collapsed',
            #         height=200
            #     )
        else:
            st.toast('Please enter a scenario')





# # Define the function to handle the 'whatif_steps' functionality
# def whatif_steps(st, scenario_steps):
#     st.divider()
        
#     st.session_state.scenario_steps = st.text_area(
#         '',
#         placeholder='Steps',
#         value=scenario_steps,
#         label_visibility='collapsed',
#         height=200
#     )
    
#     if st.button("📥 Start Agent"):
#         print('I am here')
#         if st.session_state.scenario:
#             grok_task_creation(st.session_state.scenario)
#         else:
#             st.toast('Please enter a scenario')

# Handle the main scenario text area




# # Create a container to hold the button and text input
# container = st.container()

# # Add the button and text input to the container
# with container:
#     col1, col2 = st.columns([1,3])

#     with col1:
#         st.button("Prepare Steps", key="prepare_steps_button")

#     with col2:
#         st.text_input("Scenario", key="scenario_input",label_visibility='collapsed')

# # Inject CSS to reduce the gap between the button and the text input
# st.markdown("""
# <style>
# #prepare_steps_button {
#     margin-right: 0px; /* Adjust the margin as needed */
# }
# </style>
# """, unsafe_allow_html=True)



# Load OpenAI API key from secrets
# openai.api_key = st.secrets["general"]["OPENAI_API_KEY"]

# # Function to call OpenAI's API
# def api_calling(prompt):
#     # response = openai.ChatCompletion.create(
#     #     model="gpt-3.5-turbo",
#     #     messages=[{"role": "user", "content": prompt}],
#     #     max_tokens=150,
#     #     temperature=0.7,
#     # )
    

#     llm_groq = ChatGroq(
#         model="llama-3.1-70b-versatile",
#         temperature=1,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         stop_sequences='end'
#     )    
#     resp = llm_groq.invoke(prompt)
#     return resp.content

# import streamlit as st

# with st.sidebar:
#     messages = st.container(height=300)
#     if prompt := st.chat_input("Say something"):
#         messages.chat_message("user").write(prompt)
#         messages.chat_message("assistant").write(f"Echo: {prompt}")

# # Streamlit app layout
# st.title("ChatGPT ChatBot with Streamlit")

# # Initialize session state for messages if not already done
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# # User input
# user_input = st.text_input("You: ", "")
# if user_input:
#     # Append user message to the session state
#     st.session_state.messages.append({"role": "user", "content": user_input})
    
#     # Get bot response
#     bot_response = api_calling(user_input)
    
#     # Append bot response to the session state
#     st.session_state.messages.append({"role": "assistant", "content": bot_response})

# # Display chat history
# st.write("### Chat History")
# for message in st.session_state.messages:
#     if message['role'] == 'user':
#         st.markdown(f"**You:** {message['content']}")
#     else:
#         st.markdown(f"**Bot:** {message['content']}")
        
        
# import streamlit as st
# import openai

# # Load OpenAI API key from secrets
# openai.api_key = st.secrets["general"]["OPENAI_API_KEY"]

# # Function to call OpenAI's API
# def api_calling(prompt):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=150,
#         temperature=0.7,
#     )
#     return response['choices'][0]['message']['content']

# # Streamlit app layout
# st.title("ChatGPT ChatBot with Streamlit")

# # Initialize session state for messages if not already done
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# # Create a container for chat history with scrollbar
# chat_container = st.container()
# with chat_container:
#     # Display chat history with scrollable area
#     st.write("### Chat History")
#     chat_history = st.empty()  # Placeholder for chat history

#     # Display messages in a loop
#     for message in st.session_state.messages:
#         if message['role'] == 'user':
#             st.markdown(f"**You:** {message['content']}")
#         else:
#             st.markdown(f"**Bot:** {message['content']}")

#     # Add a scrollbar to the chat history
#     st.markdown(
#         """
#         <style>
#         .streamlit-expanderHeader {
#             background-color: #f0f0f0;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# # User input at the bottom
# user_input = st.text_input("Type your message here:", "")
# if user_input:
#     # Append user message to the session state
#     st.session_state.messages.append({"role": "user", "content": user_input})
    
#     # Get bot response
#     bot_response = api_calling(user_input)
    
#     # Append bot response to the session state
#     st.session_state.messages.append({"role": "assistant", "content": bot_response})

#     # Refresh chat history display after each input
#     with chat_container:
#         for message in st.session_state.messages:
#             if message['role'] == 'user':
#                 st.markdown(f"**You:** {message['content']}")
#             else:
#                 st.markdown(f"**Bot:** {message['content']}")

# # Scroll to the bottom of the chat history on new messages
# st.write("<script>document.querySelector('div[data-testid=\"stVerticalBlock\"]').scrollTop = document.querySelector('div[data-testid=\"stVerticalBlock\"]').scrollHeight;</script>", unsafe_allow_html=True)        
