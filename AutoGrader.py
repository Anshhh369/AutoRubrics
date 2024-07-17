# Importing libraries and modules

import os
import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import StuffDocumentsChain, LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
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
from langchain.agents import AgentType, initialize_agent
from langchain.requests import Requests
from langchain_community.agent_toolkits import NLAToolkit



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
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None

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


def  get_chain(result,selected_option):

    # user_query_template = PromptTemplate(
    #     input_variables=["question" == query, "selected_option" == selected_option],
    #     template="""
    #     You are an expert in rubric generation for any given type of assignment. 
    #     Start by greeting the user respectfully, answer their {question} and verify their {selected_option}.
    #     """
    # )

    # option_selection_template = PromptTemplate(
    #     input_variables=["selected_option" == st.session_state.selected_option],
    #     template="""
    #     Collect the name from the user and then verify the {selected_option} chosen by the user.
    #     """
    # )
    
    template_1 = PromptTemplate(
        input_variables=["question" == query, "selected_option" == selected_option, "context" == result],
        template = """
        You are an expert in rubric generation for any given type of assignment. 
        Start by greeting the user respectfully, answer their {question} and verify their {selected_option}.
        Finally  based on the {selected_option}, use the persona pattern to take the persona of the  user and generate a rubric that matches their style. 
        Lastly, ask user if you want any modification or adjustments to the rubrics generated? If the user says no then end the conversation.
     
        Below is the context of how a rubric must look, use them as a reference to create detailed rubric for user.

        Context : {context}
        
        """
    )
    
    model_name = "gpt-4"
    llm = ChatOpenAI(model_name=model_name)
    
    # user_query_chain = LLMChain(llm=llm, prompt=user_query_template, verbose=True, output_key='verified_options')
    # option_selection_chain = LLMChain(llm=llm, prompt=option_selection_template, verbose=True, output_key='selected_option')
    context_based_chain = RetrievalQA.from_chain_type(llm, retriever=result.as_retriever(),chain_type_kwargs={'prompt': context_based_template})

    # sequential_chain = SequentialChain(chains=[user_query_chain, context_based_chain], input_variables=['question','selected_option', 'context'], output_variables=['verified_options','rubrics'], verbose=True)

    st.session_state.chat_active = True
    
    return context_based_chain

def python_agent():
    speak_toolkit = NLAToolkit.from_llm_and_url(llm, "https://api.speak.com/openapi.yaml")
    klarna_toolkit = NLAToolkit.from_llm_and_url(
        llm, "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"
    )
    
    natural_language_tools = speak_toolkit.get_tools() + klarna_toolkit.get_tools()

    agent_executor = create_python_agent(
        llm=llm,
        tool=natural_language_tools,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        )
    
    return agent_executor


def get_answer(query):
    chain = get_chain(st.session_state.vector_store, st.session_state.selected_option)
    answer = chain({"query": query})
    if answer == "done":
        solution = python_agent().run(
            f"Generate a rubric referring to this: {st.session_state.vector_store}, using these options: {st.session_state.selected_option}."
        )
        return solution
        
    else:

        return answer['result']

def select_option():
    
    options = ("Broad Overview", "Moderately Detailed", "Highly Detailed")
    
    if st.session_state.selected_option not in options:
        st.session_state.selected_option = options[0]
        
    selected_option = st.selectbox(
        "Detail Level of Criteria",
        options,
        index=options.index(st.session_state.selected_option)
    )
    st.write("You selected:", selected_option)
    st.session_state.selected_option = selected_option

    return selected_option


# Title for the web app
st.title("🦜🔗 AutoGrader")

# Multi-page navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Upload Document", "Ask Question"])


if page == "Home":
    st.write("Welcome to AutoGrader! Select options and use the sidebar to navigate.")
    selected_option = select_option()
    

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























    # # This controls how each document will be formatted. Specifically,
    # # it will be passed to `format_document` - see that function for more
    # # details.

    # template = """


    # You are an expert in rubric generation for any given type of assignment. 
 
    # Start by greeting the user respectfully and help them answer their {question}. 
    # Collect the name from the user and verify below information from the context. 
    
    # Context: {options}


    # """


    # document_prompt = PromptTemplate(
    #     input_variables=[options == "st.session_state.option", question == "query"],
    #     template=template
    # )
    # document_variable_name = "result"
    # # llm = OpenAI()
    # # The prompt here should take as an input variable the
    # # `document_variable_name`


    # prompt = PromptTemplate.from_template(
    #     """ 
    #     use the persona pattern to take the persona of the  user and generate a rubric that matches their style. 
    #     Lastly, ask user if you want any modification or adjustments to the rubrics generated? If the user says no then end the conversation.
    #     Below is the context of how a rubric must look, use them as a reference to create detailed rubric for user.

    #     Context : {result}
    #     """
    # )
    
    # # # Creating the Prompt
 
    # # system_prompt = """
     
    # # You are an expert in rubric generation for any given type of assignment. 
 
    # # Start by greeting the user respectfully and help them answer their {question}. 
    # # Collect the name from the user and then follow below steps:

    # # Gather the {options} selected by the user. 
    # # Finally  based on the gathered preferences, use the persona pattern to take the persona of the  user and generate a rubric that matches their style. 
    # # Lastly, ask user if you want any modification or adjustments to the rubrics generated? If the user says no then end the conversation.
     
    # # Below is the context of how a rubric must look, use them as a reference to create detailed rubric for user.

    # # Context : {context}
    
     
    # # """
    
    # # system_prompt.format(options = "inputs", context = "result", question = "query")
    
    # # prompt = ChatPromptTemplate.from_messages(
    # #     [("system", system_prompt), ("human", "{question}")]
    # # )

    
    # #Define a function to find similar documents based on a given query
     
    
    # # Assigning the OPENAI model and Retrieval chain
     
    # model_name = "gpt-4"
    # llm = ChatOpenAI(model_name=model_name)
     
    # r_chain = RetrievalQA.from_chain_type(llm, retriever=result.as_retriever(),chain_type_kwargs={'prompt': prompt}
    #                                )

    # chain = StuffDocumentsChain(
    #     llm_chain=r_chain,
    #     document_prompt=document_prompt,
    #     document_variable_name=document_variable_name
    # )
