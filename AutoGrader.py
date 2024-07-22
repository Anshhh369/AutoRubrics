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
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)


from langchain import PromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain.requests import Requests
from langchain_community.agent_toolkits import NLAToolkit
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough


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
    st.session_state.selected_option = []

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

     


def format_chat_history(messages):
    formatted_history = ""
    for message in messages:
        role = "user" if message["role"] == "user" else "Assistant"
        content = message["content"]
        formatted_history += f"{role}: {content}\n"
    return formatted_history

def  get_chain(options,context,chat_history):

    system_prompt = """
    
        You are an expert in rubric generation for any given type of assignment. 
        
        Start by greeting the user respectfully, collect the name of the user.
        The user has already selected {options} for the factors like Detail level of criteria,Grading strictness,Area of emphasis, Assignment type and Assignment style.
        Verify these selections with user by displaying the options in the following format:

        Detail Level of Criteria: 
        Grading Strictness:
        Area of Emphasis in Grading:
        Assisgnment Type:
        Assisgnment Style:

        After verifying all the options, generate a rubric referring to the format of examples and instructions provided in the {context}, make sure you use the same format.
        If there is nothing available in {context}, suggest the user to upload one for better response.
        Use the persona pattern to take the persona of the  user and generate a rubric that matches their style. 
        Lastly, ask user if you want any modification or adjustments to the rubrics generated? If the user says no then end the conversation.
        Keep the chat history to have memory and not repeat questions.
        
        chat history: {chat_history}
         
        """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    prompt.format_messages(input = "query", options = "st.session_state.selected_option", context = "st.session_state.vector_store", chat_history = "chat_history")

    model_name = "gpt-4"
    llm = ChatOpenAI(model_name=model_name)

    chain = LLMChain(llm=llm, prompt=prompt)

    if st.session_state.vector_store:
        retriever = context.as_retriever()
        chain = create_retrieval_chain(retriever, chain)

    st.session_state.chat_active = True

    return chain
    
    
    

import re

def extract_information(conversation, pattern):
    for line in conversation:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return None


def get_answer(query):
    # st.write(f"Selected Option: {st.session_state.selected_option}")
    chains = get_chain(st.session_state.selected_option,st.session_state.vector_store,chat_history)
    response = chains.invoke({"input": query, "options": st.session_state.selected_option, "context" : st.session_state.vector_store, "chat_history": chat_history})
    try:
        answer = response['text']
    except:
        # pattern = r'text:'
        # answer = extract_information(answer, pattern)
        ans = response['answer']
        answer = ans['text']
        

        
    return answer

def select_option():
    
    options_1 = ("Broad Overview", "Moderately Detailed", "Highly Detailed")
    options_2 = ("Linient","Somewhat Linient","Moderate","Very Strict")
    options_3 = ("Technical Accuracy","Depth of Analysis","Clarity and Creativity","Real-World Application","Problem Solving Skills")
    options_4 = ("Writing Report or Essay","Coding or Programming Aissignment","Design or Creative Project","Research Paper or Thesis","Case Study Analysis")
    options_5 = ("Research Oriented","Problem Solving","Case Studies","Presentations","Experiential Learning","Literature Reviews","Reflective Journals")
    
    option_1 = st.selectbox(
        "Detail Level of Criteria",
        options_1,
        index = None
    )
    st.write("You selected:", option_1)
    st.session_state.selected_option.append(option_1)

    option_2 = st.selectbox(
        "Grading Strictness",
        options_2,
        index = None
    )
    st.write("You selected:", option_2)
    st.session_state.selected_option.append(option_2)
    
    option_3 = st.selectbox(
        "Area of Emphasis in Grading",
        options_3,
        index = None
    )
    st.write("You selected:", option_3)
    st.session_state.selected_option.append(option_3)

    option_4 = st.selectbox(
        "Assignment Type",
        options_4,
        index = None
    )
    st.write("You selected:", option_4)
    st.session_state.selected_option.append(option_4)

    option_5 = st.selectbox(
        "Assignment Style",
        options_5,
        index = None
    )
    st.write("You selected:", option_5)
    st.session_state.selected_option.append(option_5)
        
    return st.session_state.selected_option


# Title for the web app
st.title("🦜🔗 AutoRubrics")

# Multi-page navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Upload Document"])


if page == "Home":
    st.write("Welcome to AutoRubrics! Select options and use the sidebar to navigate.")

    
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
    
            chat_history = format_chat_history(st.session_state.messages)
    
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
    st.session_state.uploaded_files = st.file_uploader(
        "Upload your document", type=["txt"], accept_multiple_files=True
    )
    if st.session_state.uploaded_files:
        st.session_state.vector_store = example_file(st.session_state.uploaded_files) 
    


            

                      
 


    

        

            
            
        
    
            


        























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




# def python_agent():
#     speak_toolkit = NLAToolkit.from_llm_and_url(llm, "https://api.speak.com/openapi.yaml")
#     klarna_toolkit = NLAToolkit.from_llm_and_url(
#         llm, "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"
#     )
    
#     natural_language_tools = speak_toolkit.get_tools() + klarna_toolkit.get_tools()

#     agent_executor = create_python_agent(
#         llm=llm,
#         tool=natural_language_tools,
#         verbose=True,
#         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         handle_parsing_errors=True,
#         )
    
#     solution = agent_executor.run(
#         f"""
#         Based on the: {st.session_state.selected_option}, generate a rubric referring to the context: {st.session_state.vector_store}.
#         If there is no context available, ask the user to upload one.
#         Use the persona pattern to take the persona of the  user and generate a rubric that matches their style. 
#         Lastly, ask user if you want any modification or adjustments to the rubrics generated? If the user says no then end the conversation.
#         """
#     )
    
#     return solution
