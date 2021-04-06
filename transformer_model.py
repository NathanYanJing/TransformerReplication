import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# Mask module
def create_padding_mask(src, pad=0):
    src_mask = (src == pad)
    # print(src_mask)
    # return src_mask[None, None, ...]
    return src_mask.unsqueeze(1).unsqueeze(1)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    device = torch.device("cuda")
    # mask_device = mask.to(device)
    # print(device)
    return (mask == 1).to(device)


def create_masks(src, tgt):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(src)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(src)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    dec_self_attention_look_ahead_mask = create_look_ahead_mask(tgt.shape[1])
    dec_self_attention_padding_mask = create_padding_mask(tgt)
    combined_mask = dec_self_attention_look_ahead_mask.logical_or(dec_self_attention_padding_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


"""Positional encoding module"""


def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(torch.arange(position)[:, None],
                            torch.arange(d_model)[None, :],
                            d_model)
    # apply sin to even indices in the array position 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array position 2i+1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    device = torch.device("cuda")
    return (angle_rads[None, ...]).to(device)  # 1 x


"""# Scaled dot product attention"""
# calculation at s_q, d_k level
def scaled_dot_product_attention(query, keys, values, mask=None):
    # query = (batch size->d_b, head_num -> n_h, length of query ->s_q, head_szie =>d_q)
    # keys  = (batch size->d_b, head_num -> n_h, length of keys ->s_k, head_szie =>d_k)
    # values = (batch size->d_b, head_num -> n_h, length of values ->s_v, head_szie =>d_h)
    # d_q = d_h
    d_k = keys.shape[-1]
    dot_score = query @ keys.transpose(-2, -1) / math.sqrt(d_k)  # compute dot product
    # check mask and drop-out
    if mask is not None:
        dot_score = dot_score.masked_fill(mask == 0, -1e9)
    attn_score = torch.softmax(dot_score, dim=-1)
    return attn_score @ values, attn_score



# **Multi-head Attention**


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.d_model = self.num_heads * self.depth
        # init layers
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def reshape_for_multi_heads_attention(self, t):
        # change the size to attention computation required size
        # d_model => num_heads x depth
        batch_size = t.shape[0]
        t = t.view(batch_size, -1, self.num_heads, self.depth)  # (batch_size, seq_len, num_heads, depth)
        return t.transpose(1, 2)  # (batch_size, num_heads, seq_len, depth)

    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]
        # apply linear transformations
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)
        # change the size to attention computation required size
        # d_model => num_heads x depth
        q = self.reshape_for_multi_heads_attention(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.reshape_for_multi_heads_attention(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.reshape_for_multi_heads_attention(v)  # (batch_size, num_heads, seq_len_v, depth)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v,
                                                                           mask)  # (batch_size, num_heads, seq_len_q, depth)
        # concate q, k v from different heads
        scaled_attention = scaled_attention.transpose(2, 1).contiguous().view(batch_size, -1,
                                                                              self.d_model)  # (batch_size, seq_len_q, d_model)
        return self.wo(scaled_attention)
