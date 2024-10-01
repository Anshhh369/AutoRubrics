import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, create_retrieval_chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
import re
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_community.vectorstores import AzureSearch
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
# from docx import Document 


secrets = st.secrets


openai_api_key = secrets["openai"]["api_key"]  # Accessing OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = openai_api_key  # Setting environment variable for OpenAI API key 


azure_api_key = secrets["azure"]["api_key"]
os.environ["AZURE_API_KEY"] = azure_api_key
os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = "https://ragservices.search.windows.net"


vector_store_address = "https://ragservices.search.windows.net"
vector_store_password = azure_api_key

index_name = "autorubrics-vectordb"
model = "text-embedding-ada-002"

OpenAIEmbeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model = model)


def  get_chain(options,assignment,context,chat_history):

    system_prompt = """
    
        You are an expert in rubric generation for any given type of assignment. 
        
        Start by greeting the user respectfully, collect the name of the user.
        The user has already selected {options} for the factors like Detail level of criteria, Grading strictness, Area of emphasis, Assignment type and Assignment style.
        Verify these selections with user by displaying the options in the following format:
        \n Detail Level of Criteria: 
        \n Grading Strictness:
        \n Area of Emphasis in Grading:
        \n Assignment Type:
        \n Assignment Style:

        After verifying all the options, ask the user to upload the assignment.
        Use the persona pattern to take the persona of the  user and generate a rubric for {assignment} that matches their style. 
        Make sure you refer the context given below before generating the rubric and use the same format of rubrics as given in the examples in context.
        
        Context : {context}
        
        Lastly, ask user if you want any modification or adjustments to the rubrics generated? 
        If and only if the user says no to the above question and is satisfied with the rubric then end the conversation and save the whole final generated rubric in a variable with user's name and display it in the exact following format. 

        Variable = whole final generated rubric
        

        Keep the chat history to have memory and not repeat questions and be consistent with the rubric generated.
        
        chat history: {chat_history}
         
        """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    prompt.format_messages(input = "query", assignment = "st.session_state.vector_store", options = "st.session_state.selected_option", context = "st.session_state.context", chat_history = "st.session_state.chat_history")

    model_name = "gpt-4"
    llm = ChatOpenAI(model_name=model_name)

    st.session_state.chain = LLMChain(llm=llm, prompt=prompt)


    st.session_state.chat_active = True

    return st.session_state.chain
    

def get_answer(query):
    # st.write(f"Selected Option: {st.session_state.selected_option}")
    chains = get_chain(st.session_state.selected_option,st.session_state.vector_store,st.session_state.context,st.session_state.chat_history)
    response = chains.invoke({"input" : query, "assignment": st.session_state.vector_store, "options": st.session_state.selected_option,"context": st.session_state.context,"chat_history": st.session_state.chat_history})
    
    try:
        answer = response['text']
    except:
        ans = response['answer']
        answer = ans['text']

    pattern = r"^(.*=)([\s\S]*)$"
    # for text in answer.splitlines():
    search_result = re.search(pattern, answer,re.DOTALL)
    if search_result:
        result = search_result.group()
        
        with open("extracted_information.txt", "w+") as file:                
            # Write the extracted information to the file
            file.write(result + "\n")
                
            file.seek(0)

            documents = []
            for line in file:
                document = Document(page_content=line.strip())
                documents.append(document)
                
                
                vector_store_2 = AzureSearch(
                    azure_search_endpoint=vector_store_address,
                    azure_search_key=vector_store_password,
                    index_name="predefined_rubrics",
                    api_version = "2023-11-01",
                    embedding_function=OpenAIEmbeddings.embed_query,
                    # Configure max retries for the Azure client
                    additional_search_client_options={"retry_total": 4},
                )
                
                db_2 = vector_store_2.add_documents(documents)

                if db_2:
                    st.write("Final Rubric Submitted")

                    break
    

        
    return answer






 # for line in st.session_state.messages:
 #                if isinstance(line,dict):
 #                    if 'answer' in line:
 #                        search_result = re.search(pattern,line['answer'], re.DOTALL)

 #                        if search_result: 
 #                            result = search_result.group(1)

 #                            st.write("final rubric:", result)
