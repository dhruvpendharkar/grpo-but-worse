import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

def calculate_log_probabilities(model, input_ids, outputs, tokenizer):


    full_sequence = torch.cat([input_ids, outputs], dim=-1)
    padded_sequence = pad_sequences(full_sequence, tokenizer.pad_token_id)

    with torch.no_grad():
        logits = model(padded_sequence).logits
    input_lengths = [len(input_id) for input_id in input_ids]
    answer_logits = []
    for i, input_len in enumerate(input_lengths):
        answer_logits.append(logits[i, input_len:-1, :])

    log_probs = [torch.log_softmax(answer_logit, dim=-1) for answer_logit in answer_logits]
    answer_log_probs = [
        log_prob.gather(1, output[:-1].unsqueeze(-1)).squeeze(-1)
        for log_prob, output in zip(log_probs, outputs)
    ]

    padded_log_probs = pad_sequences(answer_log_probs, pad_token_id=0)

    return padded_log_probs

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    return text, inputs, outputs

def grpo_loss(model, reference_model, input_ids, output_ids, advantages, old_log_probs, beta, epsilon, tokenizer):
    new_log_probs = calculate_log_probabilities(model, output_ids, input_ids, tokenizer)
    ref_log_probs = calculate_log_probabilities(reference_model, input_ids, output_ids, tokenizer)

    diff_exp = torch.exp(new_log_probs - old_log_probs)
    clipped_loss = torch.clamp(diff_exp, 1 - epsilon, 1 + epsilon)
    
    real_rewards = torch.min(diff_exp * advantages, clipped_loss * advantages)

    kl_log_ratio = new_log_probs - ref_log_probs
    kl_exp = torch.exp(kl_log_ratio)
    kl_loss = kl_exp - kl_log_ratio - 1
        
    loss = -torch.mean(real_rewards - beta * kl_loss)
    return loss, torch.mean(real_rewards), torch.mean(kl_loss)

def pad_sequences(sequences, pad_token_id):
    if not sequences:
        return torch.tensor([])

    # Find hte maximum length
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []

    for seq in sequences:
        if len(seq) < max_len:
            padding = torch.tensor([pad_token_id] * (max_len - len(seq)))
            padded_seq = torch.cat([seq, padding])

        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    return torch.stack(padded_sequences)

def train_with_grpo(model, old_model, reference_model, reward_model, tokenizer, optimizer, training_dataloader, iters=100, group_size=4, batch_size=2, epsilon=0.2, beta=0.01, update=10):

    pbar = tqdm.tqdm(range(iters), desc="Training GRPO")
    losses = []
    rewards = []
    old_model = old_model

    for i in pbar:
        iteration_loss = 0
        iteration_rewards = 0

        rollout_rewards = []
        rollout_log_probs = []
        rollout_inputs = []
        rollout_outputs = []

        #rollout
        for batch in training_dataloader:
            questions = batch['questions'].to(model.device)
            answers = batch['answers'].to(model.device)

            for i in range(batch_size):
                question = questions[i]
                answer = answers[i]
                group_reward = []
                for g in range(group_size):
                    text, inputs, outputs = generate(model, tokenizer, question)
                    reward = reward_model(text, answer)
                    group_reward.append(reward)
                    rollout_inputs.append(inputs)
                    rollout_outputs.append(outputs)
                rollout_rewards.append(torch.tensor(group_reward))

        # advantage calculations
        advantages = []
        for group_rewards in rollout_rewards:
            mean_reward = torch.mean(group_rewards)
            std_reward = torch.std(group_rewards)
            scaled_advantages = (group_rewards - mean_reward) / std_reward
            advantages.append(scaled_advantages)
        advantages = torch.stack(advantages)
        old_log_probs = calculate_log_probabilities(old_model, rollout_inputs, rollout_outputs, tokenizer)
    
        loss, real_rewards, kl_loss = grpo_loss(
            model, reference_model, rollout_inputs, rollout_outputs, advantages, old_log_probs, beta, epsilon, tokenizer
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration_loss += loss.item()
        iteration_rewards += real_rewards.item()
        
        iteration_loss /= len(training_dataloader)
        iteration_rewards /= len(training_dataloader)
        losses.append(iteration_loss)
        rewards.append(iteration_rewards)
        pbar.set_postfix({"Loss": losses[-1], "Rewards": rewards[-1]})
        if (i + 1) % update == 0:
            old_model.load_state_dict(model.state_dict())
            print(f"Updated old model at iteration {i + 1}")




