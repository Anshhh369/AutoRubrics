# Importing libraries and modules

import os
import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
import chardet
import os.path
import pathlib
import tempfile
from tempfile import NamedTemporaryFile
from langchain.chains import LLMChain,RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain import PromptTemplate

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

## Set up the environment
# Load secret keys

secrets = st.secrets  # Accessing secrets (API keys) stored securely

openai_api_key = secrets["openai"]["api_key"]  # Accessing OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = openai_api_key  # Setting environment variable for OpenAI API key

# Initialize session state variables
if "option" not in st.session_state:
    st.session_state.option = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Load the document, split it into chunks, embed each chunk and load it into the vector store.
def example_file(uploaded_files):
    detector = chardet.UniversalDetector()
    for uploaded_file in uploaded_files:
        # Display file details
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
        st.write(file_details)
        
        # Create temporary directory and save file there
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        with open(path, "rb") as f:
            for line in f:
                detector.feed(line)
                if detector.done:
                    break

        detector.close()
        encoding = detector.result['encoding']
         
        raw_documents = TextLoader(path,encoding = encoding).load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        db = Chroma.from_documents(documents, OpenAIEmbeddings())

    return db


def  get_chain(result):
    
    # Creating the Prompt
 
    system_prompt = """
     
    You are an expert in rubric generation for any given type of assignment. 
 
    Start by greeting the user respectfully and help them answer their {question}. 
    Collect the name from the user and then follow below steps:

    Gather the {inputs} selected by the user. 
    Finally  based on the gathered preferences, use the persona pattern to take the persona of the  user and generate a rubric that matches their style. 
    Lastly, ask user if you want any modification or adjustments to the rubrics generated? If the user says no then end the conversation.
     
    Below is the context of how a rubric must look, use them as a reference to create detailed rubric for user.

    Context : {context}
    
     
    """
    
    system_prompt.format(inputs = "st.session_state.option", context = "result", question = "query")
    
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )

    
    #Define a function to find similar documents based on a given query
     
    
    # Assigning the OPENAI model and Retrieval chain
     
    model_name = "gpt-4"
    llm = ChatOpenAI(model_name=model_name)
     
    r_chain = RetrievalQA.from_chain_type(llm, retriever=result.as_retriever(),chain_type_kwargs={'prompt': prompt}
                                   )

    st.session_state.chat_active = True
    
    return r_chain
  
def get_answer(query):
    chain = get_chain(st.session_state.vector_store)
    answer = chain({"query": query})

    return answer['result']

def select_option():
    
    options = ("Broad Overview", "Moderately Detailed", "Highly Detailed")
    
    if st.session_state.option not in options:
        st.session_state.option = options[0]
        
    option = st.selectbox(
        "Detail Level of Criteria",
        options,
        index=options.index(st.session_state.option)
    )
    st.write("You selected:", option)
    

    return option


# Title for the web app
st.title("🦜🔗 AutoGrader")

# Multi-page navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Upload Document", "Ask Question"])

if page == "Home":
    st.write("Welcome to AutoGrader! Select options and use the sidebar to navigate.")
    st.session_state.option = select_option()
    

elif page == "Upload Document": 
    st.session_state.uploaded_files = st.file_uploader(
        "Upload your document", type=["txt"], accept_multiple_files=True
    )
            
    # Button to process uploaded file
    if st.button("Process Your Files",  help = "Click to process your file before asking questions"):
        if "st.session_state.uploaded_files" is not None:
            if st.session_state.vector_store is None:
                st.session_state.vector_store = example_file(st.session_state.uploaded_files)


elif page == "Ask Question":
    if st.session_state.vector_store:
        
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
            
            # Get answer from retrieval chain
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


