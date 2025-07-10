import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import split_solution_into_chunks, get_chunk_ranges, get_chunk_token_ranges
import numpy as np
from scipy import stats

# Model and device setup
MODEL_NAME = "Qwen/Qwen2.5-14B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# The math problem prompt (repo style: uses <think> and expects \\boxed{})
problem = (
    "When the base-16 number 66666 is written in base 2, how many base-2 digits (bits) does it have?"
)
prompt = (
    "Solve this math problem step by step. Go step by step in as much detail as possible. Use many sentences, and check your work. You MUST put your final answer in \\boxed{}. "
    f"Problem: {problem} Solution: \n<think>\n"
)

GROUND_TRUTH_ANSWER = "19"  # Only the number, as extracted from \boxed{}
MAX_ATTEMPTS = 1

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
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate a chain-of-thought solution (repo-style settings)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=4096,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            do_sample=True,  # repo style: sampling
            temperature=0.9,
            top_p=0.95,
        ).sequences

    generated_ids = generated_ids[0]  # Remove batch dim if present

    # Decode the generated text
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\nGenerated CoT solution:\n", text)

    # Extract answer using repo's method
    from utils import extract_boxed_answers
    answers = extract_boxed_answers(text)
    answer = answers[0] if answers else ""
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

num_sentences = len(sentences)
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

# --- Kurtosis calculation (repo-style: vertical scores of chunk-averaged matrix) ---
def avg_matrix_by_chunk(matrix, chunk_token_ranges):
    n = len(chunk_token_ranges)
    avg_mat = np.zeros((n, n), dtype=np.float32)
    for i, (start_i, end_i) in enumerate(chunk_token_ranges):
        for j, (start_j, end_j) in enumerate(chunk_token_ranges):
            region = matrix[start_i:end_i, start_j:end_j]
            if region.size > 0:
                avg_mat[i, j] = region.mean().item()
    return avg_mat

def get_attn_vert_scores(avg_mat, proximity_ignore=4, drop_first=0):
    n = avg_mat.shape[0]
    vert_scores = []
    for i in range(n):
        vert_lines = avg_mat[i + proximity_ignore :, i]
        vert_score = np.nanmean(vert_lines) if len(vert_lines) > 0 else np.nan
        vert_scores.append(vert_score)
    vert_scores = np.array(vert_scores)
    if drop_first > 0:
        vert_scores[:drop_first] = np.nan
        vert_scores[-drop_first:] = np.nan
    return vert_scores

attn_shape = attn_weights[0].shape  # (batch, num_heads, seq, seq)
num_layers = len(attn_weights)
num_heads = attn_shape[1]
kurtosis_list = []  # List of (kurtosis, layer_idx, head_idx)



vert_scores_list = []
for layer in range(1, num_layers):
    for head in range(num_heads):
        layer_attn = attn_weights[layer][0, head].cpu().numpy()  # (seq, seq)
        avg_mat = avg_matrix_by_chunk(layer_attn, chunk_token_ranges)
        vert_scores = get_attn_vert_scores(avg_mat, proximity_ignore=4, drop_first=1)
        # Aggregate: use mean of vert_scores (ignoring NaNs)
        score = np.nanmax(vert_scores)
        vert_scores_list.append((score, layer, head))

# Sort by score descending (highest vert_scores)
vert_scores_list.sort(key=lambda x: x[0], reverse=True)

# Get the top 12 (layer, head) pairs
top_heads = vert_scores_list[:4]

print("\nTop 3 heads by kurtosis (repo-style, excluding layer 0):")
for rank, (kurt, layer_idx, head_idx) in enumerate(top_heads, 1):
    print(f"[{rank}] Layer {layer_idx}, Head {head_idx}, Kurtosis: {kurt}")
    # Compute sentence-level attention matrix for this head
    layer_attn = attn_weights[layer_idx][0, head_idx]  # (seq, seq)
    sentence_attn = torch.zeros(num_sentences, num_sentences)
    for i, (start_i, end_i) in enumerate(chunk_token_ranges):
        for j, (start_j, end_j) in enumerate(chunk_token_ranges):
            if start_i >= end_i or start_j >= end_j:
                continue
            sentence_pair_attn = layer_attn[start_i:end_i, start_j:end_j]
            if sentence_pair_attn.numel() == 0:
                continue
            avg_attn = sentence_pair_attn.mean()
            sentence_attn[i, j] = avg_attn
    print(f"Sentence-level attention matrix for layer {layer_idx}, head {head_idx} (shape: {sentence_attn.shape}):")
    print(sentence_attn[:5, :5])

# --- Rank sentences by average attention received from the top 3 heads (NaN treated as 0, lower triangular only) ---
sentence_scores = np.zeros(num_sentences)
sentence_counts = np.zeros(num_sentences)

for kurt, layer_idx, head_idx in top_heads:
    layer_attn = attn_weights[layer_idx][0, head_idx]  # (seq, seq)
    sentence_attn = torch.zeros(num_sentences, num_sentences)
    for i, (start_i, end_i) in enumerate(chunk_token_ranges):
        for j, (start_j, end_j) in enumerate(chunk_token_ranges):
            if start_i >= end_i or start_j >= end_j:
                continue
            sentence_pair_attn = layer_attn[start_i:end_i, start_j:end_j]
            if sentence_pair_attn.numel() == 0:
                continue
            avg_attn = sentence_pair_attn.mean()
            sentence_attn[i, j] = avg_attn

    # Convert to numpy and replace NaN with 0
    sentence_attn_np = sentence_attn.numpy()
    sentence_attn_np = np.nan_to_num(sentence_attn_np, nan=0.0)

    # Only consider lower triangular part (excluding diagonal)
    for j in range(num_sentences):
        # Attention paid TO sentence j (column j), from i > j
        lower_indices = np.arange(j+1, num_sentences)
        values = sentence_attn_np[lower_indices, j]
        sentence_scores[j] += values.sum()
        sentence_counts[j] += (values != 0).sum()

# Avoid division by zero
sentence_avgs = np.divide(sentence_scores, sentence_counts, out=np.zeros_like(sentence_scores), where=sentence_counts!=0)

descending_ranking = np.argsort(-sentence_avgs)  # descending order

print("\nSentence ranking by average attention received from top 3 heads (lower triangular, NaN treated as 0):")
for rank, idx in enumerate(descending_ranking, 1):
    print(f"[{rank}] Sentence {idx} (avg score: {sentence_avgs[idx]:.4f}, count: {int(sentence_counts[idx])}): {sentences[idx]}")
