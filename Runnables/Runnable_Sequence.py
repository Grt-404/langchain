from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation',
)
model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "write a joke on {topic}",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template = "explain the following joke \n {text}",
    input_variables=['text']
)
parser = StrOutputParser()

from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(template1, model, parser, template2, model, parser )

result = chain.invoke({'topic': 'pakistan'})
print(result)