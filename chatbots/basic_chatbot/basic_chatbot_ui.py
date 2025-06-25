import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from langchain_groq import ChatGroq


from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,llm):
    llm=llm
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## #Title of the app
st.title("Enhanced Q&A Chatbot With Groq Llama3")

# ## Adjust response parameter
# temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
# max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## MAin interface for user input
user_input=st.text_input("You:")

if user_input :
    response=generate_response(user_input,llm)
    st.write(response)
else:
    st.write("Please provide the user input")