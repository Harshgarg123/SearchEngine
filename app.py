import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with GroqAI"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Please answer user queries clearly."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, model, temperature, max_tokens):

    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Output parser
    output_parser = StrOutputParser()

    # Create chain
    chain = prompt | llm | output_parser

    # Generate answer
    answer = chain.invoke({'question': question})
    return answer


# ----------------- Streamlit UI -----------------
st.title("Enhanced Q&A Chatbot with Groq API")

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

# Supported Groq Models

llm_model = st.sidebar.selectbox(
    "Select AI Model",
    [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-groq-8b",
        "llama3-groq-90b",
        "mixtral-8x7b",
        "gemma2-9b-it"
    
    ]
)



temperature = st.sidebar.slider("Temperature", 0.0, 1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, value=150)

# Main chat input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    if not api_key:
        st.error("Please enter a valid Groq API key.")
    else:
        response = generate_response(user_input, api_key, llm_model, temperature, max_tokens)
        st.write(response)
else:
    st.write("Please provide the query")
