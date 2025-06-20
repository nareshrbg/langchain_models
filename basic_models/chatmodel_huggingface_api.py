from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
# Verify if the token is loaded
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not hf_token:
    print("Error: HUGGINGFACE_API_TOKEN not found in environment variables. Please set it in your .env file.")
    exit() # Exit if token is not found

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingface_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)
messages = [
    HumanMessage(
        content="What is the capital of India?"
    )
]
try:
    result = model.invoke(messages) # Pass the list of message objects
    print("\n--- Model Response ---")
    print(result.content)
except Exception as e:
    print(f"\nAn error occurred during model invocation: {e}")
