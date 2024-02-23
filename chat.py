from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf",
    )

# prompt = "Hi! What is your dog's"
# print (llm(prompt))

# prompt = "The capital of India is"
# print (llm(prompt))

# prompt = "The capital of US is"
# print (llm(prompt))

# prompt = "Tell me a story."
# for word in llm(prompt, stream=True):
#     print (word)

# prompt = "Tell me a story."
# for word in llm(prompt, stream=True):
#     print (word, sep="")

# prompt = "You are a storyteller. Recite  me a story. It should be of at least 1000 words."
# for word in llm(prompt, stream=True):
#     print (word, end="", flush=True)


def get_prompt (instruction: str) -> str:
    system = "You are an AI assitant that gives helpful answers. You answer the questions in a short and consise manner."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    print (prompt)
    return prompt


question = "Which city is the capital of India?"
for word in llm(get_prompt(question), stream=True):
    print (word, end="", flush=True)
