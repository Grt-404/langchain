from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.7}
)## temerature is a hyperparameter that controls the randomness of the model's output. A higher temperature value (e.g., 1.0) will result in more diverse and creative responses, while a lower temperature value (e.g., 0.2) will produce more focused and deterministic responses. In this case, a temperature of 0.7 is chosen to balance creativity and coherence in the generated text.
model = ChatHuggingFace(llm=llm)
result = model.invoke("who is the prime minister of india?")