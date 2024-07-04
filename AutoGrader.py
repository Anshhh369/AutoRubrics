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



# Load the document, split it into chunks, embed each chunk and load it into the vector store.
def example_file(path):
    # for file in uploaded_files:
    #     file.seek(0)  # Reset file pointer to beginning
        
    #     # Display file details
    #     file_details = {"filename": file.name, "filetype": file.type}
    #     st.write(file_details)
            
    #     # Create temporary directory and save file there
    #     temp_dir = tempfile.mkdtemp()
    #     path = os.path.join(temp_dir, file.name)
    #     st.write(path)

    detector = chardet.UniversalDetector()
    with open(path, 'rb') as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    encoding = detector.result['encoding']
    
        # with open(path, "wb") as f:
        #     f.write(file.getvalue())
            # raw_data = f.read()
            # result = chardet.detect(raw_data)
            # encoding = result['encoding']
         
    raw_documents = TextLoader(path,encoding = encoding).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = Chroma.from_documents(documents, OpenAIEmbeddings())

    return db


def  get_chain(result):
    
    # Creating the Prompt
 
    template = """
     
    You are an expert in rubric generation for any given type of assignment. 
 
    Start by greeting the user respectfully. 
    Collect the name from the user and then follow below steps:
    Verify the user input fields selected and then say something on your own like "thank you".
    Gather the {{detailLevel}},{{gradingStrictness}}, {{emphasisAreas}}, {{assignmentType}} and {{assignmentStyle}} information selected by the user. 
    Finally  based on the gathered preferences, use the persona pattern to take the persona of the  user and generate a rubric that matches their style. 
    Lastly, ask user if you want any modification or adjustments to the rubrics generated? If the user says no then end the conversation.
     
    Below is the context of how a rubric must look, use them as a reference to create detailed rubric for user.
     
    Context : {context}
        
    Human : {question}
     
    Assistant : 
     
     
     
    """
     
    prompt = PromptTemplate(
    input_variables=[context == "result", question == "query"], template=template
    )
     
    #Define a function to find similar documents based on a given query
     
    
     
     
    # Assigning the OPENAI model and Retrieval chain
     
    model_name = "gpt-4"
    llm = ChatOpenAI(model_name=model_name)
     
    chain = RetrievalQA.from_chain_type(llm, retriever=result.as_retriever(),chain_type_kwargs={'prompt': prompt}
                                   )

    return chain

def get_similiar_docs(query, k=1, score=False):
    if score:
        similar_docs = db.similarity_search_with_score(query, k=k)
    else:
        similar_docs = db.similarity_search(query, k=k)
    return similar_docs
  
def get_answer(query):
    similar_docs = get_similiar_docs(query)
    answer = chain(st.session_state.example_file())

    return answer['result']



# Title for the web app
st.title("🦜🔗 AutoGrader")


uploaded_files = st.file_uploader(
    "Upload your document", type=["txt"], accept_multiple_files=True
)
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        # Display file details
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
        st.write(file_details)
        
        # Create temporary directory and save file there
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
    
        # try:
        #     documents = example_file(temp_file_path)
        #     st.write("File processed successfully")
        #     st.write(documents)
        # except Exception as e:
        #     st.error(f"An error occurred: {e}")
        
# Button to process uploaded file
if st.button("Process Your Files",  help = "Click to process your file before asking questions"):
    if uploaded_files is None:
        st.write("Please upload a file first.")
    elif uploaded_files is not None:
        if "example_file" not in st.session_state:
            st.session_state.example_file = example_file(path)
  


# # Button to process uploaded file
# if st.button("Process Your Files",  help = "Click to process your file before asking questions"):
#     if uploaded_files is None:
#         st.write("Please upload a file first.")
#     elif uploaded_files is not None:
#         if "vector_store" not in st.session_state:
#             # Initialize vector store
#             st.session_state.vector_store = vector_db()

# if uploaded_files is not None:
        option = st.selectbox(
            "Detail Level of Criteria",
            ("Broad Overview", "Moderately Detailed", "Highly Detailed"),
            index=None,
            placeholder="Select contact method...",
        )
        st.write("You selected:", option)
    
        # Initialize chat history
        if option is None:
            st.write("Please select an option")

        elif option is not None:
            if "messages" not in st.session_state:
                st.session_state.messages = []
                
                if query := st.chat_input("Ask your question here"):
                    # Display user message in chat message container
                    with st.chat_message("user"):
                        st.markdown(query)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": query})
                    
                    # Get answer from retrieval chain
                    answer = get_answer(query)
                    result = answer["result"]
            
                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(result)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                    # Button to clear chat messages
                    def clear_messages():
                        st.session_state.messages = []
                        st.button("Clear", help = "Click to clear the chat", on_click=clear_messages)


