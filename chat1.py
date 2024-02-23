from ctransformers import AutoModelForCausalLM
from typing import List

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf",
    )


# def get_prompt (instruction: str) -> str:
#     system = "You are an AI assitant that gives helpful answers. You answer the questions in a short and consise manner."
#     prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
#     print (prompt)
#     return prompt


# question = "Which city is the capital of India?"
# for word in llm(get_prompt(question), stream=True):
#     print (word, end="", flush=True)


# question = "And which city is of the United States?"
# for word in llm(get_prompt(question), stream=True):
#     print (word, end="", flush=True)





def get_prompt (instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assitant that gives helpful answers. You answer the questions in a short and consise manner."
    prompt = f"\n\n### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print (prompt)
    return prompt


history = []
answer = ""
question = "Which city is the capital of India?"
for word in llm(get_prompt(question), stream=True):
    print (word, end="", flush=True)
    answer += word
history.append(answer)

answer = ""
question = "And which city is of the United States?"
for word in llm(get_prompt(question, history), stream=True):
    print (word, end="", flush=True)
    answer += word
history.append(answer)

question = "What about Saudi Arabia's?"
for word in llm(get_prompt(question, history), stream=True):
    print (word, end="", flush=True)
