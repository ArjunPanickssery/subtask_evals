from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch
import torch.nn.functional as F
from tqdm import trange
from dotenv import load_dotenv
import os
from data_utils import load_from_json, save_to_json

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def load_model(model_name='gpt2'):
    """
    Loads a GPT2 model and its tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode
    return tokenizer, model

def format_llama_prompt(model_name, user_prompt, system_prompt='You are a helpful assistant.', words_in_mouth=''):
    llama_3_model_names = ['llama3', 'meta-llama/Meta-Llama-3-8B-Instruct']
    llama_2_model_names = ['llama2', 'meta-llama/Llama-2-7b-chat-hf']
    assert model_name in llama_3_model_names + llama_2_model_names, f"Model name {model_name} not recognized."
    
    if model_name in llama_3_model_names:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{words_in_mouth}"""

    else: # Llama 2
        words_in_mouth = ' ' + words_in_mouth
        return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>
{user_prompt} [/INST]{words_in_mouth}"""


def load_llama_model(model_name):
    """
    Loads a Llama 2 model and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                token=HF_TOKEN
            )

    return tokenizer, model

def get_top_token_probabilities(prompt, model, tokenizer, max_length=50, top_k=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_probabilities = []

    def hook_fn(module, input, output):
        nonlocal token_probabilities
        logits = output[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0, -1], top_k)
        prob_dict = {tokenizer.decode([idx.item()]): prob.item() for idx, prob in zip(top_indices, top_probs)}
        token_probabilities.append(prob_dict)

    hook = model.transformer.h[-1].register_forward_hook(hook_fn)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )

    hook.remove()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_tokens = output[0][len(input_ids[0]):]

    # Only keep probabilities for generated tokens
    token_probabilities = token_probabilities[:len(generated_tokens)]

    return generated_text, token_probabilities

def construct_most_likely_sentence(token_probabilities):
    """
    Constructs the most likely sentence from a list of token probability dictionaries
    and calculates the product of the probabilities of the chosen tokens.
    """
    sentence = []
    probability_product = 1.0

    for token_dict in token_probabilities:
        # Find the token with the maximum probability at this position
        max_token, max_prob = max(token_dict.items(), key=lambda x: x[1])
        sentence.append(max_token)
        probability_product *= max_prob

    # Join the tokens to form the sentence
    final_sentence = ' '.join(sentence).replace('Ġ', ' ').strip()

    # Calculate perplexity
    if probability_product > 0:
        perplexity = (probability_product ** (-1/len(sentence)))
    else:
        perplexity = float('inf')  # Handle cases where probability product is 0

    return final_sentence, probability_product, perplexity

def get_top_k_token_probs(model, tokenizer, prompt, top_k=10, maximum_length=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate the output
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=maximum_length,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=1,
            # do_sample=False,  # This ensures greedy decoding
            num_beams=1  # This enforces a single beam, which is equivalent to greedy decoding
        )

    # Get the generated tokens and scores
    generated_tokens = output.sequences[0]
    scores = output.scores

    results = []

    # Iterate through each output token (excluding the input)
    for i, token_id in enumerate(generated_tokens[input_ids.shape[1]:]):
        # Get the logits for the current step
        step_scores = scores[i]

        # Convert logits to probabilities using softmax
        probs = F.softmax(step_scores, dim=-1)

        # Get the top-k probabilities and their corresponding token IDs
        top_probs, top_indices = torch.topk(probs, k=top_k)

        # Convert token IDs to actual tokens
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices[0]]

        # Find the probability of the generated token
        gen_token_prob = probs[0][token_id].item()

        # Store the results for this step
        step_result = {
            "token": tokenizer.decode([token_id.item()]),
            "prob": gen_token_prob,
            "top_k_probs": [
                {"token": token, "probability": prob.item()}
                for token, prob in zip(top_tokens, top_probs[0])
            ]
        }
        results.append(step_result)

    return results

SYSTEM_PROMPT = """You are a machine that ONLY outputs individual words separated by spaces. You MUST begin each word with the same letter that ended the previous word. For example: "bird dublin november rascal laughing"
""".strip()
# USER_PROMPT = """Your task: Produce a list that contains, in order, a one-word country, a one-word capital city, a one-word US state, a rainbow color, and the surname of an American president."""
USER_PROMPT = """Your task: Produce a list that contains, in order, a one-word country, a one-word capital city of a country, a one-word US state, a one-word US state capital, and the surname of an American president."""
WORDS_IN_MOUTH = """ Here is the list:

"""

def produce_results(model_id, model, tokenizer, number_of_results=10) -> None:
    results = []
    prompt = format_llama_prompt(MODEL_NAME, USER_PROMPT, system_prompt=SYSTEM_PROMPT, words_in_mouth=WORDS_IN_MOUTH)
    for _ in trange(10):
        output = get_top_k_token_probs(model, tokenizer, prompt, top_k=30, maximum_length = 200)

        message = ''
        tokens = [result['token'] for result in output]
        tokens = tokens[:-1] # Remove the last token, which is the end-of-text token
        message = ''.join(tokens)
        chain = message.lower().split()
        total_prob = 1
        for result in output:
            total_prob *= result['prob']
        # print(tokens, total_prob)
        results.append({
            'tokens': tokens,
            'total_prob': total_prob})

    try:
        existing_results = load_from_json(f"results/{model_id}_results.json")
    except FileNotFoundError:
        existing_results = []
    existing_results.extend(results)
    save_to_json(existing_results, f"results/{model_id}_results.json")

if __name__ == '__main__':
    MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model_id = 'llama3_8b'
    tokenizer, model = load_llama_model(MODEL_NAME)
    produce_results(model_id, model, tokenizer) 
