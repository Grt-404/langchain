from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation',
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Generate a tweet about {topic}",
    input_variables = ['topic']
)
template2 = PromptTemplate(
    template = "Generate a linkedin post about {topic}",
    input_variables = ['topic']
)
parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(template1, model1, parser),
    'linkedin': RunnableSequence(template2 | model2| parser)
}
    
)
result = parallel_chain.invoke({'topic': 'AI'})
print(result)