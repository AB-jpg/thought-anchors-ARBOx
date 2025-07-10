import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import split_solution_into_chunks, get_chunk_ranges, get_chunk_token_ranges
import numpy as np

# Model and device setup
MODEL_NAME = "Qwen/Qwen2.5-14B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# The math problem prompt (repo style: no "Answer:" at the end)
problem = (
    "When the base-16 number 66666 is written in base 2, how many base-2 digits (bits) does it have?\n"
    "Let's think step by step.\n"
)

GROUND_TRUTH_ANSWER = "Answer: 19"
MAX_ATTEMPTS = 5

def extract_answer(text):
    # Try to find the last 'Answer: <number>' or 'the answer is <number>'
    import re
    # Prefer 'Answer: <number>' at the end, but fallback to 'the answer is <number>'
    match = re.search(r'Answer:\s*(\d+)', text)
    if match:
        return f"Answer: {match.group(1)}"
    match = re.search(r'the answer is\s*(\d+)', text, re.IGNORECASE)
    if match:
        return f"Answer: {match.group(1)}"
    return None

# Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
)
model.eval()

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\nAttempt {attempt}...")
    # Tokenize input
    inputs = tokenizer(problem, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate a chain-of-thought solution
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            do_sample=True,  # ensure sampling
            temperature=0.9, # or higher if you want more randomness
        ).sequences

    generated_ids = generated_ids[0]  # Remove batch dim if present

    # Decode the generated text
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\nGenerated CoT solution:\n", text)

    # Extract answer and check
    answer = extract_answer(text)
    print(f"Extracted answer: {answer}")

    if answer == GROUND_TRUTH_ANSWER:
        print("Correct answer found!")
        break
    else:
        print("Incorrect answer, retrying...")

else:
    print("Failed to generate the correct answer after max attempts.")


# Split into sentences/chunks
sentences = split_solution_into_chunks(text)
print("\nSentences:")
for i, s in enumerate(sentences):
    print(f"[{i}] {s}")

# Get character and token ranges for each chunk
chunk_char_ranges = get_chunk_ranges(text, sentences)
chunk_token_ranges = get_chunk_token_ranges(text, chunk_char_ranges, tokenizer)

# Run model again to get attention weights for the generated sequence
full_attention_mask = torch.ones((1, generated_ids.shape[0]), device=model.device)
with torch.no_grad():
    outputs = model(
        generated_ids.unsqueeze(0),
        attention_mask=full_attention_mask,
        output_attentions=True,
        return_dict=True
    )
    attn_weights = outputs.attentions  # tuple: (num_layers, batch, num_heads, seq, seq)

# Choose layer and head
layer_idx = 36  # 0-based
head_idx = 6    # 0-based

if layer_idx >= len(attn_weights):
    raise ValueError(f"Model only has {len(attn_weights)} layers, but layer 36 was requested.")

num_sentences = len(sentences)
sentence_attn = torch.zeros(num_sentences, num_sentences)

# Aggregate token-level attention into sentence-level attention
layer_attn = attn_weights[layer_idx][0, head_idx]  # (seq, seq)
for i, (start_i, end_i) in enumerate(chunk_token_ranges):
    for j, (start_j, end_j) in enumerate(chunk_token_ranges):
        if start_i >= end_i or start_j >= end_j:
            continue
        sentence_pair_attn = layer_attn[start_i:end_i, start_j:end_j]
        if sentence_pair_attn.numel() == 0:
            continue
        avg_attn = sentence_pair_attn.mean()
        sentence_attn[i, j] = avg_attn

print(f"\nSentence-level attention matrix for layer {layer_idx}, head {head_idx} has shape: {sentence_attn.shape}")
print("Sample of the attention matrix (top-left 5x5):")
print(sentence_attn[:5, :5])

# Calculate kurtosis for the selected head's attention matrix
from scipy import stats
head_attn_matrix = layer_attn.cpu().numpy().flatten()
kurtosis = stats.kurtosis(head_attn_matrix, fisher=True, bias=True, nan_policy="omit")
print(f"Kurtosis for layer {layer_idx}, head {head_idx}: {kurtosis}")
