import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, create_retrieval_chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_community.retrievers import AzureAISearchRetriever

secrets = st.secrets


azure_api_key = secrets["azure"]["api_key"]
os.environ["AZURE_API_KEY"] = azure_api_key

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
        Make sure you refer the context given below before generating the rubric and use the same format of rubrics as given in the examples.
        
        Context : {context}
        
        Lastly, ask user if you want any modification or adjustments to the rubrics generated? If the user says no then end the conversation.
        
        Keep the chat history to have memory and not repeat questions and be consistent with the rubric generated.
        
        chat history: {chat_history}
         
        """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    prompt.format_messages(input = "query", assignment = "st.session_state.vector_store", options = "st.session_state.selected_option", context = "st.session_state.context", chat_history = "st.session_state.chat_history")

    model_name = "gpt-4"
    llm = ChatOpenAI(model_name=model_name)

    chain = LLMChain(llm=llm, prompt=prompt)
    

    if st.session_state.vector_store:
        
        retriever = AzureAISearchRetriever(
            content_key="assignment", 
            top_k=1, 
            index_name="autorubrics-vectordb",
        )
        chain = create_retrieval_chain(retriever, chain)

    st.session_state.chat_active = True

    st.session_state.chain = chain

    return st.session_state.chain
    

def get_answer(query):
    # st.write(f"Selected Option: {st.session_state.selected_option}")
    chains = get_chain(st.session_state.selected_option,st.session_state.vector_store,st.session_state.context,st.session_state.chat_history)
    response = chains.invoke({"input": query, "assignment": st.session_state.vector_store,"options": st.session_state.selected_option, "context" : st.session_state.context, "chat_history": st.session_state.chat_history})
    
    try:
        answer = response['text']
    except:
        ans = response['answer']
        answer = ans['text']
              
    return answer
