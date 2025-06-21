from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messages=[
    SystemMessage(content='Act as intelligent assistant'),
    HumanMessage(content='Texplain LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
