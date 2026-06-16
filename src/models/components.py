import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def _apply_rope(self, x, freqs_cis):
        # x: (batch_size, num_heads, seq_len, d_k)
        # freqs_cis: (seq_len, d_k) or (1, 1, seq_len, d_k)
        seq_len = x.size(2)

        freqs_cis = freqs_cis[:seq_len].to(x.device)

        x_reshaped = x.view(*x.shape[:-1], self.d_k // 2, 2)
        x_complex = torch.view_as_complex(x_reshaped)

        x_rotated = x_complex * freqs_cis
        return torch.view_as_real(x_rotated).flatten(-2)
        
    def forward(self, x, mask=None, freqs_cis=None):
        batch_size = x.size(0)
        
        # Linear projections
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply rotary positional embeddings
        if freqs_cis is not None:
            q = self._apply_rope(q, freqs_cis)
            k = self._apply_rope(k, freqs_cis)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(attn_output)
        
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.gate_linear = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU()
        )
        self.up_linear = nn.Linear(d_model, d_ff, bias=False)
        self.down_linear = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        gate_output = self.gate_linear(x)
        up_output = self.up_linear(x)
        ff_output = gate_output * up_output
        output = self.down_linear(ff_output)
        return output

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(Block, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, freqs_cis=None):
        attn_output = self.attention(self.norm1(x), mask=mask, freqs_cis=freqs_cis)
        x = x + self.dropout(attn_output)

        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)

        return x