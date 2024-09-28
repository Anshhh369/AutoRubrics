import streamlit as st
import tempfile
import os
import os.path
import pathlib
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import AzureSearch
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader

secrets = st.secrets 

openai_api_key = secrets["openai"]["api_key"]  # Accessing OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = openai_api_key  # Setting environment variable for OpenAI API key 

azure_api_key = secrets["azure"]["api_key"]
os.environ["AZURE_API_KEY"] = azure_api_key

vector_store_address = "https://ragservices.search.windows.net"
vector_store_password = azure_api_key

index_name = "autorubrics-vectordb"
model = "text-embedding-ada-002"


OpenAIEmbeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model = model)


def assignment_file(uploaded_files):
    for uploaded_file in uploaded_files:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
        # Save file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Determine file extension
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        # Load document based on its extension
        if file_extension == "pdf":
            loader = PyPDFLoader(path)
        elif file_extension == "docx":
            loader = Docx2txtLoader(path)
        # elif file_extension == "pptx":
        #     loader = UnstructuredPowerPointLoader(path)

        # Load documents and split text
        docs = loader.load()
        
        for doc in docs:
            text = doc.page_content
            st.write("file contents: \n", text)
            


        
        text_splitter =  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)

    
    return documents





