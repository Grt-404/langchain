from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation',
)

model = ChatHuggingFace(llm=llm)


#1st prompt(to get the detailed report about the topic)
template1 = PromptTemplate(
    template = "Write a Detailed Report On {topic}",
    input_variables = ["topic"]
)
#2nd prompt (summary of the report)
template2 = PromptTemplate(
    template= 'Write a 5 line summary on the following text. /n {text}',
    input_variables = ["text"]
)

prompt1 = template1.invoke({'topic': 'Blackhole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})

result2 = model.invoke(prompt2)
print(result2.content)
