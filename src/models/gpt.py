import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import Block, RMSNorm

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

        self.output_linear = nn.Linear(d_model, vocab_size, bias=False)
        self.output_linear.weight = self.token_embedding.weight

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads

        # RoPE Frequencies 
        head_dim = self.d_model // self.num_heads
        freqs = torch.arange(0, head_dim, 2) / head_dim
        freqs = 1 / (10000 ** freqs)
        t = torch.arange(self.max_seq_len)
        angles = torch.outer(t, freqs)  
        freqs_cis = torch.polar(torch.ones_like(angles), angles)  
        
        # Causal Mask 
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).unsqueeze(0).unsqueeze(0)
        
        # Register them as buffers to ensure they are moved to the correct device with the model
        self.register_buffer('freqs_cis', freqs_cis)
        self.register_buffer('mask', mask)

    def forward(self, x):
        batch_size, seq_len = x.size()
        assert seq_len <= self.max_seq_len, "Input sequence length exceeds model's maximum sequence length"

        x = self.token_embedding(x)

        # Dynamic mask size according to input sequence length
        batch_mask = self.mask[:, :, :seq_len, :seq_len]

        for block in self.blocks:
            x = block(x, mask=batch_mask, freqs_cis=self.freqs_cis)

        x = self.norm(x)
        logits = self.output_linear(x)
        
        return logits

