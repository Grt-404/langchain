 # we here utilize 2 models, 1 to generate notes about a topic, 1 to generate a quiz about the same topic, and one to merge both of them and display th result

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation',
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "generate short and simple notes from the following text \n {input_text}",
    input_variables=['input_text']

)
template2 = PromptTemplate(
    template = "generate 5 questions from the following text \n {input_text}",
    input_variables=['input_text']

)

template3 = PromptTemplate(
    template = "merge the provided notes and quiz into a single document \n notes -> {notes} and quiz-> {quiz}  ",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    {
        'notes': template1 | model1 | parser,
        'quiz' : template2 | model2 | parser
    }
)

merge_chain = template3 | model1 | parser
text = '''
The transformer architecture fundamentally changed how machines process natural languageIt replaced older sequential systems like recurrent neural networks and long short term memoryTraditional networks processed words in a strict sequence which caused slow training timesTransformers process entire sentences simultaneously by analyzing all words at the same timeThis parallel processing capability allows the model to train efficiently on massive datasetsAt the core of this innovative architecture lies the self attention mechanismSelf attention allows a word to look at other words in a sentence to gain contextThe model calculates mathematical relationships between every single token in the input textThrough these relationships the system determines which words are most relevant to each otherFor example the word bank changes meaning based on adjacent words like river or moneyThe attention mechanism dynamically weighs these connections to resolve such ambiguitiesThis process creates rich contextual embeddings that capture subtle nuances of human speechThe architecture consists of an encoder component and a decoder component working togetherThe encoder ingests the source text and transforms it into a continuous representationThe decoder uses that representation along with prior outputs to generate new text sequentiallyModern language models often use just the decoder or just the encoder depending on tasksMasked language modeling trains encoders by hiding certain words and predicting themAutoregressive generation trains decoders by predicting the very next word in a sequenceSince tokens are processed all at once the model lacks an inherent sense of orderTo solve this issue researchers introduced a technique called positional encodingPositional encodings add vectors to the word embeddings to preserve the sequence structureThese vectors use mathematical wave functions to indicate the specific position of wordsThis ensures the model knows the difference between a dog biting a man and a man biting a dogMulti head attention runs the attention mechanism multiple times completely in parallelEach individual attention head focuses on different types of relationships among the wordsSome heads might track subject verb agreements while others focus on pronouns and nounsThe outputs of these separate heads are concatenated and linearly transformed togetherLayer normalization is applied after attention blocks to stabilize the deep training processFeed forward networks further process the representations within each individual layerResidual connections bypass the attention layers to prevent the gradient from vanishing completelyThese residual paths allow gradients to flow directly through very deep network structuresThe final output layer uses a softmax function to generate probabilities for next wordsThe token with the highest probability is typically selected during the generation phaseThis entire structure scales incredibly well when given more computational power and dataLarger models with more parameters demonstrate surprising emergent capabilities over timeThey can perform complex reasoning tasks write software code and summarize vast documentsThe architecture forms the foundation of modern conversational agents and search enginesIt powers translation services sentiment analysis tools and automated content generationResearchers have also adapted this text structure to process images as patches instead of wordsThese vision transformers excel at image recognition object detection and generationThe foundational scientific paper introduced this concept with the phrase attention is all you needThis phrase highlighted that recurrence was completely unnecessary for state of the art resultsTraining these models requires massive clusters of graphics processing units working in tandemThe pre training phase exposes the model to billions of pages of unstructured web textDuring pre training the model learns grammar facts about the world and reasoning patternsA subsequent fine tuning phase aligns the model to follow specific user instructions safelyReinforcement learning from human feedback helps refine the output quality and toneDespite their immense success these models still face significant technical challengesThey require vast amounts of electricity and computational resources to train and deployThe context window limits how much text the model can consider at a single momentLonger documents require quadratic increases in computation due to the nature of self attentionResearchers are actively developing linear attention mechanisms to solve this scaling issueAnother challenge is hallucination where the model generates confident but false informationThe model predicts plausible text strings rather than cross checking facts against realityNevertheless the architecture remains the dominant paradigm in artificial intelligence researchIt has effectively bridged the gap between human communication and machine comprehensionEvery major large language model available today relies heavily on this fundamental designThe shift toward this framework marked a historical turning point in computer science historyFuture iterations will likely expand these capabilities into robotics and multimodal understandingThe journey of machine intelligence continues to unfold from this remarkable architectural milestone'''
chain = parallel_chain | merge_chain

result = chain.invoke({'input_text': text})

print(result)

chain.get_graph().print_ascii()