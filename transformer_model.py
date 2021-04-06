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

"""# Encoder"""


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.dropout1(self.multi_head_attention(x, x, x, mask))
        ffn_input = self.layernorm1(x + attention_output)
        ffn_output = self.dropout2(self.feed_forward(ffn_input))
        output = self.layernorm2(ffn_input + ffn_output)
        return output


class Encoder(nn.Module):
    def __init__(self, n_layer, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(n_layer)]

    def forward(self, x, mask):
        for layer in self.enc_layers:
            x = layer(x, mask)
            print("x", x)
        return x


"""# Decoder"""


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, padding_mask, look_ahead_mask):
        self_attention_output = self.dropout1(self.self_attention(x, x, x, look_ahead_mask))
        cross_attention_input = self.layernorm1(x + self_attention_output)

        cross_attention_output = self.dropout2(self.cross_attention(x, enc_output, enc_output, padding_mask))
        ffn_input = self.layernorm2(cross_attention_output + cross_attention_output)

        ffn_output = self.dropout3(self.feed_forward(ffn_input))
        output = self.layernorm3(ffn_input + ffn_output)
        return output


class Decorder(nn.Module):
    def __init__(self, n_layer, d_model, num_heads, d_ff, dropout=0.1):
        super(Decorder, self).__init__()
        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(n_layer)]

    def forward(self, x, enc_output, padding_mask, look_ahead_mask):
        for layer in self.dec_layers:
            x = layer(x, enc_output, padding_mask, look_ahead_mask)
        return x


"""# Transformer"""


class Transformer(nn.Module):
    def __init__(self, n_encoder_layer, n_decorder_layer, src_vocab_size, tgt_vocab_size, max_sen_len, d_model,
                 num_heads, d_ff, dropout):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_sen_len, d_model)
        self.encoder = EncoderLayer(d_model, num_heads, d_ff, dropout)
        # decoder
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.decoder = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, x, y, src_padding_mask, tgt_padding_mask, tgt_look_ahead_mask):
        src_sen_len = x.shape[1]
        tgt_sen_len = y.shape[1]
        # print(self.src_embedding(x).shape)
        # print(self.pos_encoding.shape)
        # print(self.pos_encoding[:, :src_sen_len, :].shape)
        # print(x.is_cuda)
        # print(y.is_cuda)
        # print(src_padding_mask.is_cuda)
        # print(tgt_padding_mask.is_cuda)
        # print(tgt_look_ahead_mask.is_cuda)
        enc_output = self.encoder(self.src_embedding(x) + self.pos_encoding[:, :src_sen_len, :], src_padding_mask)
        # print(self.tgt_embedding(y).shape)
        # print(self.pos_encoding[:, :tgt_sen_len, :].shape)
        dec_output = self.decoder(self.tgt_embedding(y) + self.pos_encoding[:, :tgt_sen_len, :], enc_output,
                                  tgt_padding_mask, tgt_look_ahead_mask)
        output = F.log_softmax(self.linear(dec_output), dim=-1)  # apply linear first
        return output
