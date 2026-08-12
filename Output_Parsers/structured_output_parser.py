
# some issue related to recent langchain updates is being faced

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation',
)
model = ChatHuggingFace(llm=llm)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

schema = [
    ResponseSchema(name='fact_1', description='First fact about the topic'),
    ResponseSchema(name='fact_2', description='Second fact about the topic'),   
    ResponseSchema(name='fact_3', description='Third fact about the topic')
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'black hole'})

result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)


# THE DOWNSIDE OF THIS OUTPUT PARSER IS THAT WE CANNOT PERFORM DATA VALIDATION HERE