# This just returns the input without changing anything, where is it used though??

# Suppose we are forming a sequential chain whose last step is top print something example template -> joke -> explanation, now we will only see explanation in the otput, what is I also wanna see the joke in the output?? For this we can use runnable passthrough

# we do this using the below code

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel
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

Joke_chain = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'explanation': RunnableSequence(template2, model, parser)
    }
)

final_chain = RunnableSequence(Joke_chain, parallel_chain)
result = final_chain.invoke("AI")
print(result)
final_chain.get_graph().print_ascii()