from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

text = '''

The Prince of Persia franchise features two distinct main storylines: the original 1989 platformer trilogy and the iconic 2003 Sands of Time 3D action-adventure series.   

1. The Original Game (1989)

The Premise: While the Sultan is away at war, his evil Grand Vizier, Jaffar, seizes control of Persia.   

The Conflict: Jaffar imprisons the Sultan's daughter and gives her an ultimatum: marry him within 60 minutes or be executed.   

The Journey: You play as an unnamed foreign traveler—the Princess’s true love—who has been thrown into the palace dungeons. You must escape the dungeon, navigate deadly traps (spikes, guillotine blades), defeat armed guards, merge with your magical shadow doppelgänger, and slay Jaffar before the 60-minute real-time clock runs out to save the Princess.   

2. The Sands of Time Era (2003)

The Premise: An unnamed Persian Prince claims the magical Dagger of Time during a siege on an Indian Maharaja's palace, incited by a traitorous Vizier.   

The Conflict: Tricked by the Vizier, the Prince uses the Dagger to unseal a giant Hourglass containing the Sands of Time. The released sands transform the palace inhabitants and the kingdom into horrifying sand monsters.   

The Journey: Protected by the Dagger's magic, the Prince pairs up with Farah, the captured daughter of the Maharaja. Together, they navigate the monster-infested palace using fluid parkour and time-manipulation abilities (rewinding mistakes, slowing time) to reach the Hourglass and seal the Sands back away.   

The Twist: After Farah sacrifices herself, the Prince rewinds time all the way back to before the initial invasion. He exposes the Vizier's betrayal to the Maharaja before it happens, defeats the Vizier, and saves everyone while remaining the only person who remembers the timeline that never occurred.'''

result = text_splitter.split_text(text)


# yaha pe ham saare sentences ke beech similarity nikaalenge, then un similarities ki standard deviation nikaalenge and then agar kahi bhi wo standard deviation 1 se zyada hai then we break the chunk there