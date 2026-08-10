from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions = 32) # dimensions is the size of the embedding vector. The larger the dimensions, the more information can be captured in the embedding, but it also requires more computational resources. In this case, a dimension of 32 is chosen to balance between capturing enough information and computational efficiency.

result = embedding.embed_query("I am the best")
print(str(result))