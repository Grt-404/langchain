from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(
    'UseFullImages\Resume (1).pdf'
)
# docs = loader.load()
splitter = CharacterTextSplitter(
    chunk_size=100, # number of chars in each sentence
    chunk_overlap = 0,
    separator= ' '
)

text = """
Human: Explain in simple terms, what is Doosra
AI: Sure! The "Doosra" is a special type of delivery in cricket, specifically a variation of the off-break bowled by an off-spinner. Here’s a simple breakdown:

1. **Off-Spinner's Regular Ball**: An off-spinner usually bowls the ball so that it spins from the off side (the right-hand side for a right-handed batsman) to the leg side (the left-hand side for a right-handed batsman). This is called an off-break.

2. **Doosra**: The Doosra is a deceptive delivery that spins the opposite way. Instead of spinning from off to leg, it spins from leg to off. This means it behaves like a leg-break, which is typically bowled by a leg-spinner.

3. **How It Works**: To bowl a Doosra, the off-spinner uses a different wrist and finger action to impart the opposite spin. The key is to make it look like a regular off-break to confuse the batsman.

4. **Effect on Batsman**: The batsman, expecting the ball to spin one way, is caught off guard when it spins the other way. This can lead to dismissals like bowled, LBW, or caught.

In essence, the Doosra is a trick ball that off-spinners use to outsmart batsmen by making the ball spin in the opposite direction to what the batsman expects.
Human: who is better ronaldo or messy, reply in 1 word
AI: Debatable.
Human: what did you just say?
AI: I said "Debatable." The question of whether Ronaldo or Messi is better is highly subjective and often leads to a lot of debate among football fans. Both players have their unique skills and have achieved incredible success in the sport.
Human: but I wanna hear your take, reply in one word, ronaldo or messy?
AI: Ronaldo.
Human: NOICE
AI: Glad you liked it! If you have any more questions or want to discuss further, feel free to ask!

"""
result1 = splitter.split_text(text) 
# result2 = splitter.split_documents(docs)

print(result1)

# what is chunk overlap?
# number of characters which two consecutive chuks have in common