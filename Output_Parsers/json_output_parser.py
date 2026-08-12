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

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age and city of a fictional person \n {format_instruction} ",
    input_variables = [],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

# format-instruction is a partial variable kyuki ye runtime pe fill nhi ho rha, ye usse pehle hi fill ho jaata hai, we can use it with any parser.

prompt = template.invoke({})
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)
print(type(final_result))



# we can also sue chain

chain = template | model | parser
result = chain.invoke({})
print(result)

# but the problem is that we cannot inforce our own schema here, we follow the schema the model gives us

# this problem is solved by structuresd Output parser