from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
load_dotenv()

llm = HuggingFaceEndpoint(
    
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation',
)
model = ChatHuggingFace(llm = llm)

template = PromptTemplate(
    template = "Suggest a catchy blog title about {topic}",
    input_variables=['topic']
)
parser = StrOutputParser
chain = template | model | parser
result = chain.invoke({'topic': "RGIPT and it's bad placement cell"})
print(result)