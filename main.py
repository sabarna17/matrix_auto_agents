import json
import logging
import streamlit as st  # 1.34.0
import extra_streamlit_components as stx
import time
import tiktoken
import urllib

from datetime import datetime
from streamlit_float import *
# from openai import OpenAI  # 1.30.1
from openai import AzureOpenAI  # 1.30.1

logger = logging.getLogger()
logging.basicConfig(encoding="UTF-8", level=logging.INFO)

st.set_page_config(page_title="What-IF Agent", page_icon="💡")

cookie_manager = stx.CookieManager(key="cookie_manager")

if "conversation_title" not in st.session_state:
    st.session_state["conversation_title"] = ''

col1, col2 = st.columns([1,3])

with col1:
    col1.title("What-IF?")
with col2:
    col2.text_input('',placeholder='Enter a scenario')
    # st.text_input("Enter a scenario")
    
# st.title("What-IF: ") #Past Conversations

float_init()

# Secrets to be stored in /.streamlit/secrets.toml
# OPENAI_API_ENDPOINT = "https://xxx.openai.azure.com/"
# OPENAI_API_KEY = "efpgishhn2kwlnk9928avd6vrh28wkdj" (this is a fake key 😉)

# To be used with standard OpenAI API
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# To be used with standard Azure OpenAI API
import os
from langchain_groq import ChatGroq 
from groq import Groq

os.environ['GROQ_API_KEY'] = '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'

# task_prompt = """
#     create 
# """
from multi_agent_v2 import call_agent, grok_task_creation
def api_calling_task(prompt):
    resp = grok_task_creation(prompt)
    # print(resp)
    return resp
    # llm_groq = ChatGroq(
    #     model="llama-3.1-70b-versatile",
    #     temperature=1,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     stop_sequences='end'
    # )    
    # resp = llm_groq.invoke(prompt)
    # return resp.content

def api_calling(prompt):
    llm_groq = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        stop_sequences='end'
    )    
    resp = llm_groq.invoke(prompt)
    return resp.content

# client = AzureOpenAI(
#     azure_endpoint=st.secrets["OPENAI_API_ENDPOINT"],
#     api_key=st.secrets["OPENAI_API_KEY"],
#     api_version="2024-02-15-preview",
# )


# This function logs the last question and answer in the chat messages
def log_feedback(icon):
    # We display a nice toast
    st.toast("Thanks for your feedback!", icon="👌")

    # We retrieve the last question and answer
    last_messages = json.dumps(st.session_state["messages"][-2:])

    # We record the timestamp
    activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "

    # And include the messages
    activity += "positive" if icon == "👍" else "negative"
    activity += ": " + last_messages

    # And log everything
    logger.info(activity)


@st.dialog("🎨 Upload a picture")
def upload_document():
    st.warning(
        "This is a demo dialog window. You need to process the file afterwards.",
        icon="💡",
    )
    picture = st.file_uploader(
        "Choose a file", type=["jpg", "png", "bmp"], label_visibility="hidden"
    )
    if picture:
        st.session_state["uploaded_pic"] = True
        st.rerun()
        



def get_conversation_title():
    if st.session_state["conversation_title"]:
        conversation_title = st.session_state["conversation_title"]
    else:
        print('-----------------------------')
        print(st.session_state["messages"])
        print('-----------------------------')
        full_text = "".join([str(item["content"]) for item in st.session_state["messages"]])
        prompt = "Summarize the following conversation in 3 words: " + full_text
        conversation_title = api_calling(prompt)
        print(conversation_title)
        
    # llm_groq = ChatGroq(
    #         model="llama-3.1-70b-versatile",
    #         temperature=1,
    #         max_tokens=None,
    #         timeout=None,
    #         max_retries=2,
    #         stop_sequences='end'
    #     )    
    # conversation_title = llm_groq.stream(st.session_state["messages"])
    
    # response = client.chat.completions.create(
    #     model=st.session_state["openai_model"],
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "Summarize the following conversation in 3 words:"
    #             + full_text,
    #         },
    #     ],
    #     stop=None,
    # )
    # conversation_title = response.choices[0].message.content

    return conversation_title


if "uploaded_pic" in st.session_state and st.session_state["uploaded_pic"]:
    st.toast("Picture uploaded!", icon="📥")
    del st.session_state["uploaded_pic"]

# Model Choice - Name to be adapted to your deployment
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-35-turbo"

# Adapted from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
if "messages" not in st.session_state:
    st.session_state["messages"] = []



user_avatar = "👩‍💻"
assistant_avatar = "🤖"

# We rebuild the previous conversation stored in st.session_state["messages"] with the corresponding emojis
for message in st.session_state["messages"]:
    with st.chat_message(
        message["role"],
        avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
    ):
        st.markdown(message["content"])

# A chat input will add the corresponding prompt to the st.session_state["messages"]
if prompt := st.chat_input("How can I help you?"):

    st.session_state["messages"].append({"role": "user", "content": prompt})

    # and display it in the chat history
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)


# If the prompt is initialized or if the user is asking for a rerun, we
# launch the chat completion by the LLM
if prompt or ("rerun" in st.session_state and st.session_state["rerun"]):

    with st.chat_message("assistant", avatar=assistant_avatar):
        # llm_groq = ChatGroq(
        #     model="llama-3.1-70b-versatile",
        #     temperature=1,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        #     stop_sequences='end'
        # )    
        # # resp = llm_groq.stream(st.session_state["messages"])
        # st.write_stream(llm_groq.stream(st.session_state["messages"]))
        stream = api_calling_task(st.session_state["messages"])
        print(stream)
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state["messages"]
        #     ],
        #     stream=True,
        #     max_tokens=300,  # Limited to 300 tokens for demo purposes
        # )
    # st.write(stream)
    st.session_state["messages"].append({"role": "assistant", "content": stream})

    # In case this is a rerun, we set the "rerun" state back to False
    if "rerun" in st.session_state and st.session_state["rerun"]:
        st.session_state["rerun"] = False

st.write("")

# If there is at least one message in the chat, we display the options
if len(st.session_state["messages"]) > 0:

    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = "history_" + datetime.now().strftime("%Y%m%d%H%M%S")

    action_buttons_container = st.container()
    action_buttons_container.float(
        "bottom: 7.2rem;background-color: var(--default-backgroundColor); padding-top: 1rem;"
    )

    # We set the space between the icons thanks to a share of 100
    cols_dimensions = [7, 20, 14.5, 9.1, 9, 8.6]
    cols_dimensions.append(100 - sum(cols_dimensions))
    col0, col1, col2, col3, col4, col5, col6 = action_buttons_container.columns(
        cols_dimensions
    )
    # col0, col1, col2, col3, col4, col5, col6, col7 = action_buttons_container.columns(
    #     cols_dimensions
    # )

    with col1:

        if st.button("📥 Start Agent"):
            st.session_state["messages"] = []
            del st.session_state["conversation_id"]

            if "uploaded_pic" in st.session_state:
                del st.session_state["uploaded_pic"]

            st.rerun()


        # And the corresponding Download button
        # st.download_button(
        #     label="📥 Save!",
        #     data=json_messages,
        #     file_name="chat_conversation.json",
        #     mime="application/json",
        # )
        

        # # Converts the list of messages into a JSON Bytes format
        # json_messages = json.dumps(st.session_state["messages"]).encode("utf-8")

        # # And the corresponding Download button
        # st.download_button(
        #     label="📥 Save!",
        #     data=json_messages,
        #     file_name="chat_conversation.json",
        #     mime="application/json",
        # )

    with col2:

        # We set the message back to 0 and rerun the app
        # (this part could probably be improved with the cache option)
        if st.button("Clear 🧹"):
            st.session_state["messages"] = []
            del st.session_state["conversation_id"]

            if "uploaded_pic" in st.session_state:
                del st.session_state["uploaded_pic"]

            st.rerun()

    with col3:

        if st.button("🎨"):
            upload_document()

    with col4:
        icon = "🔁"
        if st.button(icon):
            st.session_state["rerun"] = True
            st.rerun()

    with col5:
        icon = "👍"

        # The button will trigger the logging function
        if st.button(icon):
            log_feedback(icon)

    with col6:
        icon = "👎"

        # The button will trigger the logging function
        if st.button(icon):
            log_feedback(icon)

    # with col7:
    #     label = 'test'
    #     # # We initiate a tokenizer
    #     # enc = tiktoken.get_encoding("cl100k_base")

    #     # # We encode the messages
    #     # tokenized_full_text = enc.encode(
    #     #     " ".join([item["content"] for item in st.session_state["messages"]])
    #     # )

    #     # # And display the corresponding number of tokens
    #     # label = f"💬 {len(tokenized_full_text)} tokens"
    #     st.link_button(label, "https://platform.openai.com/tokenizer")

else:

    # At the first run of a session, we temporarly display a message
    if "disclaimer" not in st.session_state:
        with st.empty():
            for seconds in range(3):
                st.warning(
                    "‎ You can click on 👍 or 👎 to provide feedback regarding the quality of responses.",
                    icon="💡",
                )
                time.sleep(1)
            st.write("")
            st.session_state["disclaimer"] = True

st.sidebar.header("Past Conversations")
sc1, sc2 = st.sidebar.columns((6, 1))
history_keys = [key for key in reversed(list(st.context.cookies)) if key.startswith('history')]
for key, conversation_id in enumerate(history_keys):

    content = json.loads(urllib.parse.unquote(st.context.cookies.get(conversation_id)))
    title = list(content.keys())[0]
    if sc1.button(title, key=f"c{key}"):
        st.session_state["conversation_title"] = title
        st.text_input()
        
        # conversation_title = st.session_state["conversation_title"]
        # print(content)
        # st.session_state["messages"] = content
        print('----------------------')
        print(content[title])
        print('----------------------')
        st.session_state["messages"] = content[title]
        
        for message in st.session_state["messages"]:
            with st.chat_message(
                message["role"],
                avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
            ):
                # st.session_state["messages"] = 
                st.markdown(message["content"])
        
        st.sidebar.info(f'Reload "{title}"', icon="💬")

    if sc2.button("❌", key=f"x{key}"):
        st.sidebar.info("Conversation removed", icon="❌")
        cookie_manager.delete(conversation_id)

if "conversation_id" in st.session_state:
    conversation_title = get_conversation_title()
    cookie_manager.set(st.session_state["conversation_id"], val={conversation_title: st.session_state["messages"]})

