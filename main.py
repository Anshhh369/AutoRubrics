import streamlit as st
import os
import re
from app.document_processing import assignment_file
from app.example_file import example
from app.chain_response import get_chain, get_answer
from app.chat_history import format_chat_history
from app.utils import select_option
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3



# Initialize session state variables
if "selected_option" not in st.session_state:
    st.session_state.selected_option = ["None Selected"] * 5
    
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "context" not in st.session_state:
    st.session_state.context = None



# Title for the web app
st.title("🦜🔗 AutoRubrics")

# Multi-page navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Upload Document"])


if page == "Home":
    st.write("Welcome to AutoRubrics! Select options and use the sidebar to navigate.")

    st.session_state.conext = example()
    
    st.session_state.selected_option = select_option()
        
    if st.session_state.selected_option:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                                
        if query := st.chat_input("Ask your question here"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(query)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
    
            st.session_state.chat_history = format_chat_history(st.session_state.messages)
    
            answer = get_answer(query)
        
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
            # Add assistant response to chat history                
            st.session_state.messages.append({"role": "assistant", "content": answer})

            
            # Button to clear chat messages
            def clear_messages():
                st.session_state.messages = []
            st.button("Clear", help = "Click to clear the chat", on_click=clear_messages)


if page == "Upload Document":
    
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.vector_store = assignment_file(st.session_state.uploaded_files)
        if st.session_state.vector_store:
            st.write("Documents uploaded successfully.")
    


