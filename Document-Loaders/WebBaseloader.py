# used to load and extract text content from web pages

# uses beutifull soup

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(web_paths=('https://www.amazon.in/gp/product/B0CLZVTGQ1/',))  # can pass list of url's here as well

docs = loader.load()

print(docs[0].page_content)