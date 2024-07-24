import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, create_retrieval_chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough



def  get_chain(options,assignment,context,chat_history):

    system_prompt = """
    
        You are an expert in rubric generation for any given type of assignment. 
        
        Start by greeting the user respectfully, collect the name of the user.
        The user has already selected {options} for the factors like Detail level of criteria, Grading strictness, Area of emphasis, Assignment type and Assignment style.
        Verify these selections with user by displaying the options in the following format:

        Detail Level of Criteria: 
        Grading Strictness:
        Area of Emphasis in Grading:
        Assignment Type:
        Assignment Style:

        After verifying all the options, ask the user to upload the assignment and generate a rubric referring to that {assignment}.
        Make sure you learn from {context} about what makes a good rubric and use the same format as given in examples while generating the rubric.
        

        Use the persona pattern to take the persona of the  user and generate a rubric that matches their style. 
        Lastly, ask user if you want any modification or adjustments to the rubrics generated? If the user says no then end the conversation.
        Keep the chat history to have memory and not repeat questions.
        
        chat history: {chat_history}
         
        """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    prompt.format_messages(input = "query", assignment = "st.session_state.vector_store", options = "st.session_state.selected_option", context = "st.session_state.context", chat_history = "st.session_state.chat_history")

    model_name = "gpt-4"
    llm = ChatOpenAI(model_name=model_name)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    retriever = assignment.as_retriever()
    
    chain = create_retrieval_chain(retriever, llm_chain)

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
