import streamlit as st
from app.document_processing import example_file
from app.Chain_Response import get_chain, extract_information
from app.Chat_history import format_chat_history
from app.utils import select_option


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
