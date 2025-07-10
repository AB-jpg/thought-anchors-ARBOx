# interactive_demo.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import split_solution_into_chunks, get_chunk_ranges
import circuitsvis as cv
from whitebox_analyses.model_read import ActivationCollector
import numpy as np
from IPython.display import display
import gc

# Should I eat eggs or beef if I care about the environment?

# # Python Example
# from circuitsvis.tokens import colored_tokens
# colored_tokens(["My", "tokens"], [0.123, -0.226])

def get_model_and_tokenizer(model=None, tokenizer=None):
    """Helper function to get model and tokenizer instances."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
    if model is None or tokenizer is None:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True).to("cuda")
            print("Loaded new model instance")
    else:
        print("Using pre-loaded model instance")
    
    return model, tokenizer

# 1) Load model and tokenizer
model, tokenizer = get_model_and_tokenizer()

# 2) Get a prompt from the user and generate a chain of thought
prompt = "Should I eat eggs or beef if I care about suffering?" #input("Prompt: ")
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

# Create attention mask (1 for real tokens, 0 for padding)
attention_mask = torch.ones_like(input_ids)

# First generate without attention outputs
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,  # Use EOS as PAD since DeepSeek doesn't have PAD
        return_dict_in_generate=True
    ).sequences

# Now do a forward pass to get attention patterns
full_attention_mask = torch.ones((1, generated_ids.shape[1]), device=model.device)

with torch.no_grad():
    outputs = model(
        generated_ids,
        attention_mask=full_attention_mask,
        output_attentions=True,
        return_dict=True
    )
    attn_weights = outputs.attentions

text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
sentences = split_solution_into_chunks(text)
print("\nGenerated sentences:", sentences)

# 3) Get token ranges for each sentence
chunk_ranges = get_chunk_ranges(text, sentences)
print("\nToken ranges for each sentence:", chunk_ranges)

# 4) Compute per-head sentence-sentence matrices
num_layers = len(attn_weights)
num_heads = attn_weights[0].shape[1]  # Each layer's attention has shape (batch, heads, seq, seq)
print(f"\nNumber of layers: {num_layers}, Number of heads per layer: {num_heads}")
num_sentences = len(sentences)

# Initialize sentence-level attention tensor
sentence_attn = torch.zeros(num_layers, num_heads, num_sentences, num_sentences)

# Add diagnostic counters
total_nan_count = 0
nan_locations = []
empty_range_count = 0

# Aggregate token-level attention into sentence-level attention
for layer in range(num_layers):
    layer_attn = attn_weights[layer][0]  # Remove batch dimension
    for head in range(num_heads):
        head_attn = layer_attn[head]
        
        # For each pair of sentences, average the attention weights between their tokens
        for i, (start_i, end_i) in enumerate(chunk_ranges):
            for j, (start_j, end_j) in enumerate(chunk_ranges):
                # Check for empty ranges
                if start_i >= end_i or start_j >= end_j:
                    empty_range_count += 1
                    print(f"\nEmpty range detected at layer {layer}, head {head}, sentences {i}->{j}")
                    print(f"Range i: {start_i}->{end_i}, Range j: {start_j}->{end_j}")
                    print(f"Sentence i: {sentences[i]}")
                    print(f"Sentence j: {sentences[j]}")
                    continue

                # Extract and average attention weights for this sentence pair
                sentence_pair_attn = head_attn[start_i:end_i, start_j:end_j]
                avg_attn = sentence_pair_attn.mean()
                
                # Check for NaN
                if torch.isnan(avg_attn):
                    total_nan_count += 1
                    nan_locations.append({
                        'layer': layer,
                        'head': head,
                        'sentence_i': i,
                        'sentence_j': j,
                        'range_i': (start_i, end_i),
                        'range_j': (start_j, end_j),
                        'attention_shape': sentence_pair_attn.shape
                    })
                
                sentence_attn[layer, head, i, j] = avg_attn

print("\nDiagnostic Information:")
print(f"Total NaN values: {total_nan_count}")
print(f"Empty ranges encountered: {empty_range_count}")

if total_nan_count > 0:
    print("\nDetailed NaN Analysis:")
    for loc in nan_locations:
        print(f"\nNaN at layer {loc['layer']}, head {loc['head']}, sentences {loc['sentence_i']}->{loc['sentence_j']}")
        print(f"Range i: {loc['range_i']}, Range j: {loc['range_j']}")
        print(f"Attention submatrix shape: {loc['attention_shape']}")
        print(f"Source sentence: {sentences[loc['sentence_i']]}")
        print(f"Target sentence: {sentences[loc['sentence_j']]}")

print("\nAttention tensor shape:", sentence_attn.shape)

# Compute statistics about the non-NaN attention values
valid_attention = sentence_attn[~torch.isnan(sentence_attn)]
if len(valid_attention) > 0:
    print("\nAttention Statistics (excluding NaN):")
    print(f"Mean attention: {valid_attention.mean().item():.4f}")
    print(f"Min attention: {valid_attention.min().item():.4f}")
    print(f"Max attention: {valid_attention.max().item():.4f}")
    print(f"Std attention: {valid_attention.std().item():.4f}")

# 5) For each layer and head, show a heatmap with circuitsvis
# for layer in range(num_layers):
#     for head in range(num_heads):
#         print(f"\nLayer {layer}, Head {head} Attention Pattern:")
#         heatmap = cv.attention.attention_patterns(
#             sentences,
#             sentence_attn[layer, head].cpu().numpy()
#         )
#         display(heatmap)

# 5) Optionally compute causal importance matrix using compute_step_importance_matrix
