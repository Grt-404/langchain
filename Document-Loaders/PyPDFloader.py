# when you try to load a 25 page pdf it will brea that pdf into a list of of 25 document objects 
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(
    'Resume (1).pdf',
)
docs = loader.load()
print(len(docs))
# since there were two pages in the pdf i got a list of lenght 2


