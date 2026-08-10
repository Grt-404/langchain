from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions = 32) # dimensions is the size of the embedding vector. The larger the dimensions, the more information can be captured in the embedding, but it also requires more computational resources. In this case, a dimension of 32 is chosen to balance between capturing enough information and computational efficiency.

documents = [
    "virat kohli is the best batsman in the world",
    "sachin tendulkar is the worst batsman in the world",
    "ronaldo is the best footballer in the world",
    "messi is the worst footballer in the world",
]

query = "who is the best footballer in the world"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding) # need to pass 2d lists

index, score = sorted(list(enumerate(scores)), key =lambda x:x[1], reverse=True) #sorting the scores in descending order and returning the index of the document with the highest score

print(query)
print(documents[index])
print(score)