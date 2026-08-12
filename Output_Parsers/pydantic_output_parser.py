from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation',
)

model = ChatHuggingFace(llm=llm)

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description= ' Name of the person') 
    age: int = Field(gt=18, description = 'age of the person')
    city: str = Field(description=' Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object= Person)
template = PromptTemplate(
    template = "Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}

)
prompt = template.invoke({'place': "Indian"})
result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)

print(prompt)

