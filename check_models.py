import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

models = [
    'microsoft/Phi-3.5-mini-instruct',
    'Qwen/Qwen2.5-0.5B-Instruct',
    'google/gemma-2-2b-it',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'meta-llama/Llama-3.2-1B-Instruct',
    'Qwen/Qwen2.5-3B-Instruct',
]

for repo in models:
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo,
            task='text-generation',
            huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN'),
        )
        out = llm.invoke('Say hello in one word.')
        print(repo, 'OK', out)
    except Exception as e:
        print(repo, 'ERR', type(e).__name__, e)
