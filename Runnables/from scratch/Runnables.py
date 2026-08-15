import random

# AAM ZINDAGI(WITHOUT RUNNABLE):

# class NakliLLm:
#     def __init__(self):
#         print('llm created')
#     def predict(self, prompt):
#         response_list = [
#             'Delhi is the capital of India',
#             'IPL is a cricket league',
#             'AI stands for Artificial Intelligence'
#         ]
#         return {'response': random.choice(response_list)}

# class NakliPromptTemplate:
#     def __init__(self, template, input_variables):
#         self.template = template
#         self.input_variables = input_variables

#     def format(self, input_dict):
#         return self.template.format(**input_dict)

# class NakliLLMChain:
#     def __init__(self, prompt, llm):
#         self.prompt = prompt
#         self.llm = llm

#     def run(self, input_dict):
#         final_prompt= self.prompt.format(input_dict)
#         result = self.llm.predict(final_prompt)

#         print(result['response'])

# llm = NakliLLm()
# template = NakliPromptTemplate(
#     template = 'Write a poem on {topic}', 
#     input_variables=['topic']
# )
# chain = NakliLLMChain(template, llm)
# chain.run({'topic': 'india'})


# here they realized their mistake
# and concept of runnable was born using abstraction


# MENTOS ZIDAGI(WITH RUNNABLES)
from abc import ABC, abstractmethod
class Runnable(ABC):

    def invoke(self, input_dict):
        return self.template.format(**input_dict)
    # jo kaam format kr rha tha whi same kaam ib invoke krega har class mei so that there is a structure mantained
    @abstractmethod
    def invoke(input_data):
        pass
# Abstract Classes: Base classes that define abstract methods (methods without a body/implementation) that subclasses must implement.
class NakliPromptTemplate(Runnable):

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
      

    def format(self, input_dict):
        return self.template.format(**input_dict)


class StrOutputParser(Runnable):
    def __init__(self):
        pass
    def invoke(self, inputDict):
        return(inputDict['response'])

parser = StrOutputParser()
class NakliLLm(Runnable):
    def __init__(self):
        print('llm created')

    def invoke(self, prompt):
        response_list = [
                    'Delhi is the capital of India',
                    'IPL is a cricket league',
                    'AI stands for Artificial Intelligence'
                ]
        return {'response': random.choice(response_list)}

    def predict(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]
        return {'response': random.choice(response_list)}

class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
    # suppose ki pehla runnable is a template class object, wo invoke hua to uska prompt ban gaya , and then wo prompt jab loop agli baar chalega then next runnable ke invoke ka  input variable ban jayega
    def invoke(self, input_data):
        for runnable in self.runnable_list:
           input_data =  runnable.invoke(input_data)
        return input_data

template = NakliPromptTemplate(
    template = 'Give me a poem on {topic}',
    input_variables=['topic']
)

llm = NakliLLm()

chain = RunnableConnector([template, llm, parser])
result = chain.invoke({'topic': 'india'})
print(result)

# connecting multiple chains

template2 = NakliPromptTemplate(
    template = "generate the explanation of the {response}",
    input_variables = ['response']
)

chain1 = RunnableConnector([template, llm])
chain2 = RunnableConnector([template2, llm, parser])

chain3 = RunnableConnector([chain1, chain2])
result = chain3.invoke({'topic': 'china'})

print(result)