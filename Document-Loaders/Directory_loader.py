# A type of document loader that lets you load multiple documents from a directpry of files

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
loader = DirectoryLoader(
    path= "UseFullImages",
     glob= "**/*.{txt,pdf}" 
    #  **/*  recursively searches through subdirectories.
#     When you use a normal single-asterisk pattern like *.txt, Python only looks inside the immediate folder you specified. It ignores any folders sitting inside that folder.

# Adding ** tells Python to search down through every level of nested subfolders inside that main folder.

# .{txt,pdf} uses brace expansion to target both file extensions at once.
)


#PROBLEM

# using the directpry loader we are loading all files at once to the memory, and load one mei time bhi boht lag rha tha

 # SOLUTION
 # LAZY LOAD

docs = loader.lay_load()

for document in docs:
    print(document.metadata)

    # we need not wait for all of them to be processed it will be procecced on by one and then thrown out of the memory and processed again 