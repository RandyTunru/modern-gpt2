import os
import math
import torch
import torch.nn.functional as F
import wandb
from src.utils.metrics import ThroughputMonitor

class Trainer:
    def __init__(self, model, dataloader, optimizer, config, device, start_step=0):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.start_step = start_step
        
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.monitor = ThroughputMonitor(config['batch_size'] * self.grad_accum_steps, config['max_seq_len'])

    def get_lr(self, step):
        max_lr = self.config['learning_rate']
        min_lr = max_lr * 0.1  # Usually 10% of max_lr
        warmup_steps = self.config.get('warmup_steps', 1000)
        total_steps = self.config['max_steps']

        # 1. Linear warmup
        if step < warmup_steps:
            return max_lr * (step / warmup_steps)
        
        # 2. Minimum LR after training exceeds max_steps
        if step > total_steps:
            return min_lr
            
        # 3. Cosine decay down to min_lr
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        
        return min_lr + coeff * (max_lr - min_lr)

    def train(self):
        self.model.train()

        step = self.start_step
        
        ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        
        data_iter = iter(self.dataloader)
        
        print("Starting training loop...")
        self.monitor.start()

        while step < self.config['max_steps']:
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            
            # --- Gradient Accumulation Loop ---
            for micro_step in range(self.grad_accum_steps):
                try:
                    X, Y = next(data_iter)
                except StopIteration:
                    if self.config['no_epochs']:
                        print("Dataset exhausted. Stopping training loop.")
                        return
                    data_iter = iter(self.dataloader)
                    X, Y = next(data_iter)

                X, Y = X.to(self.device), Y.to(self.device)

                with ctx:
                    logits = self.model(X)

                    # Reshape for CrossEntropyLoss: x (batch_size * seq_len, vocab_size) and y (batch_size * seq_len)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        Y.view(-1)
                    )

                    # Scale loss based on gradient accumulation steps to maintain correct optimization dynamics
                    loss = loss / self.grad_accum_steps
                
                # Backward Pass
                loss.backward()
                accum_loss += loss.item() # Accumulate the scaled loss for logging purposes

            # --- Optimization Step ---
            # Gradient clipping to prevent exploding gradients (once per optimization step, not per micro-step)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Step the optimizer
            self.optimizer.step()

            # Logging & Checkpointing
            if step % self.config['log_every'] == 0:
                tps = self.monitor.get_tps()
                wandb.log({
                    "train/loss": accum_loss,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "metrics/tokens_per_sec": tps,
                    "step": step
                })
                print(f"Step {step:05d} | Loss: {accum_loss:.4f} | TPS: {tps:.0f}")

            if step > 0 and step % self.config['save_every'] == 0:
                self.save_checkpoint(step)

            step += 1

    def save_checkpoint(self, step):
        filepath = os.path.join(self.checkpoint_dir, f"model_step_{step}.pt")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")