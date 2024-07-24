import chardet
import tempfile
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader

def assignment_file(uploaded_files):
    detector = chardet.UniversalDetector()
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
            st.write("file contents:", text)
            


        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)
        db = Chroma.from_documents(documents, OpenAIEmbeddings())
        
    return db





        with open(path, "rb") as f:
            for line in f:
                detector.feed(line)
                if detector.done:
                    break
        detector.close()
        encoding = detector.result['encoding']
        raw_documents = TextLoader(path, encoding=encoding).load()
