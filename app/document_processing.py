import streamlit as st
import tempfile
import os
import os.path
import pathlib
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader
import pdfplumber
from langchain.docstore.document import Document
import docx

secrets = st.secrets 

openai_api_key = secrets["openai"]["api_key"]  # Accessing OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = openai_api_key  # Setting environment variable for OpenAI API key 

azure_api_key = secrets["azure"]["api_key"]
os.environ["AZURE_API_KEY"] = azure_api_key

os.environ["AZURE_AI_SEARCH_API_KEY"] = azure_api_key
os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = "https://ragservices.search.windows.net"

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
            with pdfplumber.open(uploaded_file) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
                text = "\n".join(pages)
        elif file_extension == "docx":
            loader = Docx2txtLoader(path)
            pages = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in pages.paragraphs])


        # Load documents and split text
        docs = loader.load()
        
        for doc in docs:
            content = doc.page_content
            # st.write("file contents: \n", content)
        
        text_splitter =  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)

        vector_store = AzureSearch(
            azure_search_endpoint=vector_store_address,
            azure_search_key=vector_store_password,
            index_name=index_name,
            api_version = "2023-11-01",
            embedding_function=OpenAIEmbeddings.embed_query,
            # Configure max retries for the Azure client
            additional_search_client_options={"retry_total": 4},
        )
        db = vector_store.add_documents(documents)

        if db:

            query = uploaded_file.name

            # st.write("name: ", query)
            
            files = vector_store.similarity_search(
                query=query,
                k=1000, 
                search_type="similarity"
            )

            content = []
            for file in files:
                
                document = file.page_content
                content.append(document)
                
            st.write("Assignment: ", content)

            return content

    
    return docs





