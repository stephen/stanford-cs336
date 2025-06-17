d_model = 1600
vocab_size = 50257
num_layers = 48
context_length = 1024
num_heads = 25
d_ff = d_model * 4

batch_size = 2

parameter_count = ((2*d_model + 4*d_model**2 + 3*d_model*d_ff) * num_layers + d_model + (d_model * vocab_size) + d_model)

activation_count_rmsnorm = d_model
activation_count_attention = 3 * batch_size * context_length + context_length * (d_model / num_heads) + batch_size * context_length **2

activation_count_attention_kqv = 3 * batch_size * context_length * d_model + context_length**2
activation_count_attention_mask = context_length * context_length
activation_count_attention_qk_and_softmax = batch_size * 2 * context_length**2
activation_count_attention_v = batch_size * context_length * d_model
activation_count_attention = activation_count_attention_kqv + activation_count_attention_mask + activation_count_attention_qk_and_softmax + activation_count_attention_v

activation_count_mlp = batch_size * d_ff * 3

activation_count_layers = (activation_count_attention + activation_count_mlp + activation_count_rmsnorm) * 3
activation_count_unembedding = vocab_size * batch_size
activation_count_softmax = batch_size * d_model
activation_count = activation_count_layers * 48 + activation_count_unembedding + activation_count_softmax + activation_count_rmsnorm

gradient_count = parameter_count
optimizer_state_count = parameter_count * 2

print(f"{parameter_count}, {activation_count * 4}, {gradient_count}, {optimizer_state_count}")

mem_bytes = 4 * (parameter_count + activation_count + gradient_count + optimizer_state_count)
print(f"{mem_bytes} bytes")
print(f"{mem_bytes / 1024} MB")
print(f"{mem_bytes / 1024 / 1024} GB")
print(f"{mem_bytes / 1024 / 1024 / 1024} GB")

print("fixed cost", (parameter_count *4 *4 / 1024 / 1024 / 1024), "GB")
# print("batch-based cost", activation_count * )


