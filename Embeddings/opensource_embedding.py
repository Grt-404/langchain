import os

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # sentence-transformers/all-MiniLM-L6-v2 is a pre-trained model that can be used to generate embeddings for text data. It is a smaller version of the MiniLM model, which is designed to be efficient and fast while still providing good performance for many NLP tasks. The "L6" in the model name indicates that it has 6 layers, which is a relatively small number of layers compared to larger models like BERT or GPT-3. The "v2" indicates that this is the second version of the model, which may include improvements or updates over the first version.
text = "I am the best"
result = embedding.embed_query(text)
print(str(result))

# here we run it locally we can also do it using the inference as follows
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    endpoint_url="https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN", "YOUR_HUGGINGFACE_TOKEN"),
)  # sentence-transformers/all-MiniLM-L6-v2 is a pre-trained model that can be used to generate embeddings for text data. It is a smaller version of the MiniLM model, which is designed to be efficient and fast while still providing good performance for many NLP tasks. The "L6" in the model name indicates that it has 6 layers, which is a relatively small number of layers compared to larger models like BERT or GPT-3. The "v2" indicates that this is the second version of the model, which may include improvements or updates over the first version.
text = "I am the best"
result = embedding.embed_query(text)
print(str(result))