import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Load .env variables
load_dotenv()

# Streamlit settings
st.set_page_config(page_title="AI Chatbot")
st.title("🤖 AI Chatbot using LangChain")

# Ensure API key is present
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Missing GOOGLE_API_KEY in .env file!")
    st.stop()

# Initialize LangChain Model (Using the active 2026 model: gemini-2.5-flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Chat history using LangChain message formats
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    # Check if the message is from the user or the AI
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# User input
prompt = st.chat_input("Type your message")

if prompt:
    # Show user message in UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message to LangChain memory
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Generate response via LangChain
    try:
        # Pass the entire conversation history to the model
        response = llm.invoke(st.session_state.messages)
        
        # Show bot response in UI
        with st.chat_message("assistant"):
            st.markdown(response.content)

        # Save bot response to LangChain memory
        st.session_state.messages.append(AIMessage(content=response.content))
        
    except Exception as e:
        st.error(f"An API error occurred: {e}")