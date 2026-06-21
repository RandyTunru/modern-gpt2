import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import Block, RMSNorm

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, original_seq_len=None):
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

        self.original_seq_len = original_seq_len if original_seq_len is not None else max_seq_len

        # RoPE Frequencies 
        head_dim = self.d_model // self.num_heads
        freqs = torch.arange(0, head_dim, 2) / head_dim
        freqs = 1 / (10000 ** freqs)

        # Scale factor to adjust the RoPE frequencies based on the original sequence length
        scale_factor = self.max_seq_len / self.original_seq_len

        t = torch.arange(self.max_seq_len) / scale_factor # Divide by scale_factor to adjust the effective sequence length for RoPE

        angles = torch.outer(t, freqs)  
        freqs_cis = torch.polar(torch.ones_like(angles), angles)  
        
        # Causal Mask 
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).bool().unsqueeze(0).unsqueeze(0)
        
        # Register them as buffers to ensure they are moved to the correct device with the model
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        self.register_buffer('mask', mask, persistent=False)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()
        assert seq_len <= self.max_seq_len, "Input sequence length exceeds model's maximum sequence length"

        x = self.token_embedding(x)

        # Dynamic mask size according to input sequence length
        batch_mask = self.mask[:, :, :seq_len, :seq_len]

        if attention_mask is not None:
            batch_mask = batch_mask & attention_mask

        for block in self.blocks:
            x = block(x, mask=batch_mask, freqs_cis=self.freqs_cis)

        x = self.norm(x)
        logits = self.output_linear(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=0, repetition_penalty=1.0):
        self.eval() 
        
        for _ in range(max_new_tokens):
            # If current sequence is longer than max_seq_len, take only the last max_seq_len tokens as input to the model
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            logits = self(idx_cond) # Pass the current sequence to the model

            logits = logits[:, -1, :] # Focus on the last token's logits for generation
            
            if repetition_penalty != 1.0:
                # Iterate over the batch dimension
                for b in range(logits.shape[0]):
                    # Find all unique tokens that have appeared in this sequence so far
                    prev_tokens = torch.unique(idx[b])
                    
                    # Fetch the current logits for those specific tokens
                    prev_logits = logits[b, prev_tokens]
                    
                    # Apply the penalty: multiply if negative, divide if positive
                    penalized_logits = torch.where(
                        prev_logits < 0, 
                        prev_logits * repetition_penalty, 
                        prev_logits / repetition_penalty
                    )
                    
                    # Assign the penalized logits back to the main logits tensor
                    logits[b, prev_tokens] = penalized_logits
            
            # Apply temperature scaling
            logits = logits / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if idx_next.item() == eos_id: # Stop generation if the model generates the end-of-sequence token
                break

        self.train() # Revert model back to training mode after generation
        return idx

