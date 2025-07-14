from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def build_reward_prompt(question: str, correct_answer: str, candidates: list[str]) -> str:

    prompt = f"""Here is a question and the correct answer:

Question: {question}
Correct Answer: {correct_answer}

Below are several candidate answers. Please rank them from best to worst, considering how well they match the correct answer.

"""
    for idx, cand in enumerate(candidates, start=1):
        prompt += f"Candidate {idx}: {cand.strip()}\n"

    prompt += "\nPlease output the ranking as a list of integers, e.g., [3,1,2,...] where 1 is best.\n"
    return prompt


def parse_ranking_output(output_text: str, group_size: int) -> list[int]:

    import re

    # Look for something like [3,1,2]
    match = re.search(r"\[(.*?)\]", output_text)
    if not match:
        raise ValueError(f"Could not parse ranking from: {output_text}")
    
    numbers_str = match.group(1)
    numbers = [int(n.strip()) for n in numbers_str.split(",")]
    if len(numbers) != group_size:
        raise ValueError(f"Expected {group_size} rankings, got {len(numbers)}")
    
    return numbers


@torch.no_grad()
def get_rewards_from_ranking(model, tokenizer, question: str, correct_answer: str, candidates: list[str], group_size: int, device="cpu") -> list[float]:
 
    prompt = build_reward_prompt(question, correct_answer, candidates)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False
    )

    output_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(output_text)
    ranks = parse_ranking_output(output_text, group_size)  # list of ints, e.g., [3,1,2]

    # Map ranks to rewards: best rank = 1.0, worst = 0.0, linear scale
    rewards = [1.0 - (rank-1)/(group_size-1) for rank in ranks]

    return rewards