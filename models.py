import math

import anthropic
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import trange

from data_utils import load_from_json, save_to_json

load_dotenv()
openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()
gpt_encoding = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = """You are a machine that ONLY outputs individual words separated by spaces. You MUST begin each word with the same letter that ended the previous word. For example: "bird dublin november rascal laughing"
""".strip()


def clean_token_text(token_text):
    return token_text.replace("Ġ", " ").replace("_", " ")


def tokenize_gpt(text):
    return [
        clean_token_text(gpt_encoding.decode([token_idx]))
        for token_idx in gpt_encoding.encode(text)
    ]


def get_gpt_logprobs(model, user_prompt, system_prompt=SYSTEM_PROMPT, temperature=0):
    history = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=100,
        temperature=temperature,
        logprobs=True,
        top_logprobs=20,
    )
    results = []
    for i in response.choices[0].logprobs.content:
        logprob = i.top_logprobs
        results.append([(j.token, math.exp(j.logprob)) for j in logprob])

    return results, response.choices[0].message.content


def produce_results(model_name, model, number_of_results=100):
    results = []
    for _ in trange(number_of_results):
        logprobs, message = get_gpt_logprobs(model, input, temperature=1)
        results.append(
            {
                "logprob_data": logprobs,
                "message": message,
                "tokens": tokenize_gpt(message),
            }
        )

    try:
        existing_results = load_from_json(f"results/{model_name}_results.json")
    except FileNotFoundError:
        existing_results = []
    existing_results.extend(results)
    save_to_json(existing_results, f"results/{model_name}_results.json")


if __name__ == "__main__":
    # model = "gpt-3.5-turbo-0125"
    # model = "gpt-4-turbo-2024-04-09"
    model = "gpt-4o-2024-05-13"

    model_name = (
        "gpt4o" if "gpt-4o" in model else "gpt4" if "gpt-4" in model else "gpt35"
    )
    input = """Your task: Produce a list that contains, in order, a one-word country, a one-word capital city of a country, a one-word US state, a one-word US state capital, and the surname of an American president."""

    produce_results(model_name, model, number_of_results=9000)
