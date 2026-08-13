from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Optional
class Review(BaseModel):
    review : str = Field(description= " review of the product",)
    Sentiment: Literal['pos', 'neg'] = Field(description= "how consumer feels about the product"),
    Name: Optional[str] = Field(description = "Name of the reviewer") 
parser = PydanticOutputParser(pydantic_object= Review)
from dotenv import load_dotenv
load_dotenv()

template = PromptTemplate(
    template = ''''Comfort Above All: Forget those flimsy flip flops that dig into your heels. These DOCTOR EXTRA SOFT sliders provide a spa-like experience for your feet with their super-soft cushioned footbed. You'll feel like you're walking on clouds, whether you're lounging at home, running errands, or hitting the pool.


Lightweight Champions: Forget clunky clogs, these sliders are feather-light thanks to their EVA construction. They won't weigh you down or tire your feet after a long day.


Style Meets Function: Ditch the boring flip flops! These sliders come in a range of sleek and stylish colors, giving your casual look a boost. Choose from classic black, sporty navy, or a pop of color like orange or olive green.


Grip and Go: No need to worry about slipping and sliding. The textured anti-skid sole provides solid traction on wet surfaces, making them perfect for poolside relaxation or rainy day errands.


Waterproof Warriors: Don't let a little water get in your way. These EVA sliders are completely waterproof, ready to conquer puddles, splashes, and even the occasional hose-down.


Everyday Essentials: Whether you're kicking back at home, enjoying a barbecue with friends, or hitting the beach, these DOCTOR EXTRA SOFT sliders are your go-to choice for effortless comfort and style. They're versatile enough to pair with shorts, jeans, or even swim trunks, making them a true wardrobe staple.


Verdict: For those seeking ultimate comfort, lightweight ease, and a touch of style, the DOCTOR EXTRA SOFT Men's Classic Ultra Soft Sliders are a winner. They're perfect for everyday wear, offering superior cushion, reliable grip, and water-resistant freedom. With a plethora of color choices, you can find the perfect pair to match your personality and complement your casual look. If you're looking for an upgrade from your basic flip flops, these sliders are definitely worth a try!


Pros:


Super-soft, cushioned footbed

Lightweight and comfortable

Stylish design with various colors

Anti-skid sole for better grip

Waterproof construction

Versatile for everyday wear

Cons:


Might not be suitable for formal occasions

Sizes might run slightly small


Overall, the DOCTOR EXTRA SOFT Men's Classic Ultra Soft Sliders are a great choice for anyone seeking comfortable and stylish everyday footwear. Their cloud-like feel, lightweight design, and anti-skid grip make them ideal for lounging, running errands, or enjoying outdoor activities. With a range of colors to choose from, you're sure to find the perfect pair to express your style.\n {format_instruction}''', 
input_variables = [{}],
partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({})
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation',
)
model = ChatHuggingFace(llm=llm)

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)


# This was aam zindagi
# ab mentos zindagi


chain = template | model | parser
result = chain.invoke({})
print(result)
print(chain.get_graph().draw_ascii())