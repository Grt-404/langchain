# to convert a python function to a runnable
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda
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

parser = StrOutputParser()

Joke_chain = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'no_of words': RunnableLambda(lambda x : len(x.split()))
    }
)

final_chain = RunnableSequence(Joke_chain, parallel_chain)
result = final_chain.invoke("AI")
print(result)
final_chain.get_graph().print_ascii()