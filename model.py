"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    curr_idx = 0
    vocab_dict = {}

    for sp in specials:
        vocab_dict[sp] = curr_idx
        curr_idx += 1

    for sentence in sentences:
        for word in sentence.split(" "):
            if word not in vocab_dict:
                vocab_dict[word] = curr_idx
                curr_idx += 1
    
    return vocab_dict

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    id_to_token = {v: k for k, v in token_to_id.items()}
    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    if not len(sentence):
        return []
    ids = [token_to_id[token if token in token_to_id else unk_token] for token in sentence.split(" ")]
    return ids

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    sentence = [id_to_token[tid] for tid in ids]
    return sentence

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    paded_seq = [pad_id] * max_len
    for i in range(min(len(ids), max_len)):
        paded_seq[i] = ids[i]

    return paded_seq

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    padded_seq_tensor = torch.tensor(padded_sequences, dtype=torch.int64)
    return padded_seq_tensor

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    
    return embeddings * (d_model**(1/2))

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    divisors = torch.tensor([10000**(-2*i/d_model) for i in range(0, d_model//2)])
    return divisors

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    
    return torch.tensor([[i] for i in range(0, max_len)], dtype=torch.float32)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # for i in range(0, pe.shape[1], 2):
    #     pe[:, i] = torch.sin(position.squeeze(-1) * div_term[i//2])
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    division_term = compute_positional_div_term(d_model)
    position = build_position_index_column(max_len)

    pe = torch.zeros(max_len, d_model)

    pe = fill_even_indices_with_sin(pe, position, division_term)
    pe = fill_odd_indices_with_cos(pe, position, division_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    B, L, d_model = embedded_batch.shape
    pos_added_batch = embedded_batch[::, :, :] + positional_encoding[:L, :]
    return pos_added_batch

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    bool_pad_mask = ~(token_ids==pad_id)
    return bool_pad_mask.unsqueeze(1).unsqueeze(2)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    rows, cols = torch.meshgrid(torch.arange(seq_len), torch.arange(seq_len), indexing='ij')
    mask = rows >= cols

    # INFO
    # mat = np.ones(seq_len, seq_len)
    # mask = torch.tril(mat, diagonal=1)

    return mask.unsqueeze(0).unsqueeze(1)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    return padding_mask[::, :, :, :] & causal_mask[:, :, ::, :]

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    key_dim = key.dim()
    return query @ key.transpose(key_dim-1, key_dim-2)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    return scores / (d_k**(1/2))

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    return torch.where(mask, scores, float("-inf"))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    weights = torch.softmax(masked_scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    return weights

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""

    d_k = key.shape[-1]
    L_k = key.shape[-2]

    attn_scores = compute_raw_attention_scores(query, key)
    attn_scores = scale_attention_scores(attn_scores, d_k)

    if mask is not None:
        attn_scores = mask_attention_scores_with_neg_inf(attn_scores, mask)

    attn_weights = softmax_attention_weights(attn_scores)
    context = apply_attention_weights_to_values(attn_weights, value)

    return context, attn_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    B, L, d_model = tensor.shape
    return tensor.reshape(B, L, num_heads, d_model//num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    return split_tensor.permute(0, 2, 1, 3)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    multi_head_tensor_permute = multi_head_tensor.permute(0, 2, 1, 3)
    B, L, num_heads, d_k = multi_head_tensor_permute.shape
    return multi_head_tensor_permute.reshape(B, L, num_heads*d_k)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    return x @ weight.t() + (0 if bias is None else bias)

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)

    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))

    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    merge_context = merge_heads_back_to_model_dim(context)
    o = apply_linear_projection(merge_context, w_o, b_o)
    
    return o

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    Q = apply_linear_projection(query, w_q, None)
    K = apply_linear_projection(key, w_k, None)
    V = apply_linear_projection(value, w_v, None)

    q_h, k_h, v_h = split_qkv_into_heads(Q, K, V, num_heads)

    context, attn_weights = scaled_dot_product_attention(q_h, k_h, v_h, mask)

    O = merge_heads_and_project_output(context, w_o, None)
    
    return O

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    o = apply_linear_projection(x, w1.t(), b1)
    z = torch.nn.functional.relu(o)
    
    return z

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    o = apply_linear_projection(hidden, w2.t(), b2)
    return o

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)
    o = apply_ffn_second_linear(hidden, w2, b2)

    return o

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # population correction, by default variance is divided by N-1, correction 0 makes it divide by N
    return x.mean(dim=-1).unsqueeze(-1), x.var(dim=-1, correction=0).unsqueeze(-1)

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, var = compute_layer_norm_mean_and_variance(x)

    x_hat = (x - mean) / (var + eps)**(1/2)
    
    return x_hat * gamma + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    x = residual_input + sublayer_output
    x_norm = normalize_and_scale_with_gamma_beta(x, gamma, beta, eps)

    return x_norm

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    
    return x*keep_mask / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    x_attn_ctx = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    o = apply_residual_add_and_norm(x, x_attn_ctx, gamma, beta)

    return o

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    x_ffn =  position_wise_feed_forward_network(x, w1, b1, w2, b2)
    o = apply_residual_add_and_norm(x, x_ffn, gamma, beta)
    return o

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    x_attn_enc = encoder_layer_self_attention_sublayer(
        x, 
        layer_params['w_q'], layer_params['w_k'], layer_params['w_v'], layer_params['w_o'], 
        layer_params['attn_gamma'], layer_params['attn_beta'],
        num_heads, src_mask)

    o = encoder_layer_feed_forward_sublayer(
        x_attn_enc, 
        layer_params['w1'], layer_params['b1'], 
        layer_params['w2'], layer_params['b2'], 
        layer_params['ffn_gamma'], layer_params['ffn_beta'])
    
    return o

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    for i, layer_params in enumerate(encoder_layer_params_list):
        x = assemble_encoder_layer(x, layer_params, num_heads, src_mask)
    
    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 44 - decoder_layer_cross_attention_sublayer (not yet solved)
# TODO: implement

# Step 45 - decoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 46 - assemble_decoder_layer (not yet solved)
# TODO: implement

# Step 47 - stack_decoder_layers (not yet solved)
# TODO: implement

# Step 48 - apply_final_output_projection (not yet solved)
# TODO: implement

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

