from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# detailed way
template2 = PromptTemplate(
    template='Define {topic} in max 2 sentence',
    input_variables=['topic']
)

# fill the values of the placeholders
prompt = template2.invoke({'topic':'AI'})

result = model.invoke(prompt)

print(result.content)