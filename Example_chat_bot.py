# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-72B-Instruct",
#     task = 'text-generation',
# )
# model = ChatHuggingFace(llm=llm)

# result = model.invoke("how are you?")
# print(result.content)



# from langchain_core.prompts import PromptTemplate
# template = PromptTemplate(
#     template = "Explain the {research_paper} in {Type} way",
#     input_variables = ["research_paper","Type"]
#  )
# prompt = template.invoke({"research_paper":input("Enter your topic"),"Type":"Mathematical"})
# result = model.invoke(prompt.to_string())
# print(result.content)


# designing a chatbot
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation',
)
model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{user_input}')
])

CHAT_FILE = "chat.txt"
chat_history = []

if os.path.exists(CHAT_FILE):
    with open(CHAT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Human: "):
                chat_history.append(HumanMessage(content=line[7:]))
            elif line.startswith("AI: "):
                chat_history.append(AIMessage(content=line[4:]))

domain = input("Enter expert domain: ")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == "exit":
        break
    
    if not user_input:
        continue

    prompt = chat_template.invoke({
        'domain': domain,
        'user_input': user_input,
        'chat_history': chat_history
    })

    result = model.invoke(prompt)
    ai_response = result.content
    print(f"\nAI: {ai_response}\n")

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=ai_response))

    with open(CHAT_FILE, "a", encoding="utf-8") as f:
        f.write(f"Human: {user_input}\n")
        f.write(f"AI: {ai_response}\n")