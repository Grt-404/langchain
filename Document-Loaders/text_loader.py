from langchain_community.document_loaders import TextLoader
loader = TextLoader('chat.txt', encoding = 'utf-8')

docs = loader.load()

print(type(docs))  # All document loaders load documents as list of document

print(len(docs))

print(type(docs[0]))


# it will have page_content, metadata
print(docs[0].page_content)

(docs[0].metadata)