# Importing libraries and modules

import os
import re
import openai
import chardet
import os.path
import pathlib
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain, create_retrieval_chain
from langchain.chains import LLMChain,RetrievalQA
from langchain.prompts.chat import ChatPromptTemplate
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
    
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None



# Load the document, split it into chunks, embed each chunk and load it into the vector store.
def example_file(uploaded_files):
    detector = chardet.UniversalDetector()
    for uploaded_file in uploaded_files:
        # Display file details
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
        
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
    


def get_answer(query):
    # st.write(f"Selected Option: {st.session_state.selected_option}")
    chains = get_chain(st.session_state.selected_option,st.session_state.vector_store,chat_history)
    response = chains.invoke({"input": query, "options": st.session_state.selected_option, "context" : st.session_state.vector_store, "chat_history": chat_history})
    
    try:
        answer = response['text']
    except:
        ans = response['answer']
        answer = ans['text']
              
    return answer


def format_chat_history(messages):
    formatted_history = ""
    for message in messages:
        role = "user" if message["role"] == "user" else "Assistant"
        content = message["content"]
        formatted_history += f"{role}: {content}\n"
    return formatted_history


def select_option():
    
    options_1 = ("Broad Overview", "Moderately Detailed", "Highly Detailed")
    options_2 = ("Linient","Somewhat Linient","Moderate","Very Strict")
    options_3 = ("Technical Accuracy","Depth of Analysis","Clarity and Creativity","Real-World Application","Problem Solving Skills")
    options_4 = ("Writing Report or Essay","Coding or Programming Aissignment","Design or Creative Project","Research Paper or Thesis","Case Study Analysis")
    options_5 = ("Research Oriented","Problem Solving","Case Studies","Presentations","Experiential Learning","Literature Reviews","Reflective Journals")
    
 # Maintain index based on previous selections if they exist
    option_1 = st.selectbox(
        "Detail Level of Criteria",
        options_1,
        index=options_1.index(st.session_state.selected_option[0]) if len(st.session_state.selected_option) > 0 else 0
    )
    st.session_state.selected_option[0:1] = [option_1]

    option_2 = st.selectbox(
        "Grading Strictness",
        options_2,
        index=options_2.index(st.session_state.selected_option[1]) if len(st.session_state.selected_option) > 1 else 0
    )
    st.session_state.selected_option[1:2] = [option_2]
    
    option_3 = st.selectbox(
        "Area of Emphasis in Grading",
        options_3,
        index=options_3.index(st.session_state.selected_option[2]) if len(st.session_state.selected_option) > 2 else 0
    )
    st.session_state.selected_option[2:3] = [option_3]

    option_4 = st.selectbox(
        "Assignment Type",
        options_4,
        index=options_4.index(st.session_state.selected_option[3]) if len(st.session_state.selected_option) > 3 else 0
    )
    st.session_state.selected_option[3:4] = [option_4]

    option_5 = st.selectbox(
        "Assignment Style",
        options_5,
        index=options_5.index(st.session_state.selected_option[4]) if len(st.session_state.selected_option) > 4 else 0
    )
    st.session_state.selected_option[4:5] = [option_5]

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
    
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.vector_store = example_file(st.session_state.uploaded_files)
        st.write("Documents uploaded successfully.")
    
    if st.session_state.uploaded_files:
        for uploaded_file in st.session_state.uploaded_files:
            st.write("Uploaded files:", uploaded_file.name) 
    

