# this is not working because this model is maybe not trained to return Structured Output

#there is no data validation anywhere, suppose you said age is a number but model returns string, then it will not throw any error, it will just return the string. So you have to validate the data yourself or use pydantic

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(

    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation',


)
model = ChatHuggingFace(llm=llm)

#schema
# JUST THE SCHEMA IN JSON FORMAT LIKE THE SAMPLE WE CREATED
json_schema= {
    "title": "Student",
    "desciption": "Schema about student",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Name of the student"
        },
        "age": {
            "type": "integer",
            "description": "Age of the student"
        }
    },
    "required": [
        "name"
    ]
}

structured_model = model.with_structured_output(json_schema)
result = model.invoke("""Hi I have been using this product for 3months now. good results wrt hot flashes and sleeplessness. However, I have few questions to do product owners.

1. you have mentioned 2 tablets of multivitamin per day. can I take 2 tablets in same time?

2. You have mentioned serving is 1550mg nutrition. Is it 1550mg per tablet or for 2 tablets?

3. the nutrition details given are for one tablet or for 2 tablets??

4. in multivitamin already vitamin D3 has 600IU. if I take a combo of both multivitamin and Calcium, will it not be overdose?

without knowing these details, I have been taking 2 tablets of both multivitamin and calcium. not sure if I am overdosing myself. Please provide proper information. 4 rating is for product but need to have right information which is missing.""")
print(result.content)