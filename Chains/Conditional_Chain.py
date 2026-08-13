# suppose a user gives a feed back, if sentiment is positive, I need to give a differet repy, and if negative , we need to generate a different response.

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


from typing import Annotated, Literal
from pydantic import BaseModel, Field
model = ChatHuggingFace(llm=llm)
class Review(BaseModel):
    Sentiment: Literal['Positive', 'Negative'] = Field(description= "sentiment of the review")


parser2 = PydanticOutputParser(pydantic_object= Review)
template1 = PromptTemplate(
    template = " classify the sentiment of the following feedback text into positive or negative \n {feedback} {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

parser1 = StrOutputParser()
classifier_chain = template1 | model | parser2

# branch_chain = RunnableBranch(
#     (condition1, act to perform is condition 1 is true),
#     (condition2, act to perform is condition 2 is true),
#     default(if none of the mentioned conditions are true)
# )
template2 = PromptTemplate(
    template = "write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)
template3 = PromptTemplate(
    template = "write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    (lambda x: x.Sentiment == 'Positive', template2 | model |parser1),
    (lambda x: x.Sentiment == 'Negative', template3 | model |parser1),
    RunnableLambda(lambda x: "could not find sentiment")  # as we could not just use that fucnction becuse we need to provide a  chain in the default case, Runnable LAmbda kinda converts it into a chain

)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'this is a terrible phone'})
print(result)

chain.get_graph().print_ascii()