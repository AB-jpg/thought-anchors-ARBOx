import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import split_solution_into_chunks, get_chunk_ranges, get_chunk_token_ranges
import numpy as np
from scipy import stats

# Model and device setup
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# The math problem prompt (repo style: uses <think> and expects \\boxed{})
problem = input("Prompt:")

prompt = problem


# Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
)
model.eval()

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate a chain-of-thought solution (repo-style settings)
with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1024,
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
for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        layer_attn = attn_weights[layer_idx][0, head_idx].cpu().numpy()  # (seq, seq)
        avg_mat = avg_matrix_by_chunk(layer_attn, chunk_token_ranges)
        vert_scores = get_attn_vert_scores(avg_mat, proximity_ignore=4, drop_first=0)
        kurt = stats.kurtosis(vert_scores, fisher=True, bias=True, nan_policy="omit")
        kurtosis_list.append((kurt, layer_idx, head_idx))

# Exclude layer 0 from kurtosis analysis
kurtosis_list = [entry for entry in kurtosis_list if entry[1] != 0]

# Sort by kurtosis descending and take top 3
kurtosis_list.sort(reverse=True, key=lambda x: x[0])
top_heads = kurtosis_list[:3]

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
