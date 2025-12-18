import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler




api_wrapper_wikipedia = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)

wikipedia=WikipediaQueryRun(api_wrapper=api_wrapper_wikipedia)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)


ddg_search=DuckDuckGoSearchRun(name="Search")


st.title("Langchain Chatbot + Search Engine")




############### Sidebar

st.sidebar.title("API Key")
api_key=st.sidebar.text_input("Enter your Groq API Key: ",type="password")


##############


if "messages" not in st.session_state:

    st.session_state["messages"]=[{
        "role":"assistant",
        "content":"Hi, I'm a chatbot and I can search the web. How can I help you?"
    }]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])



if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model='llama-3.1-8b-instant',streaming=True)
    tools=[ddg_search,arxiv,wikipedia]
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION

    search_agent=initialize_agent(tools,llm,agent=agent,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[streamlit_callback])
        st.session_state.messages.append({
            "role":"assistant",
            "content":response
        })
        st.write(response)