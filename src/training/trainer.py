# src/training/trainer.py
import os
import torch
import torch.nn.functional as F
import wandb
from src.utils.metrics import ThroughputMonitor

class Trainer:
    def __init__(self, model, dataloader, optimizer, config, device):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.monitor = ThroughputMonitor(config['batch_size'], config['max_seq_len'])

    def train(self):
        self.model.train()
        step = 0
        
        ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        
        data_iter = iter(self.dataloader)
        
        print("Starting training loop...")
        self.monitor.start()

        while step < self.config['max_steps']:
            try:
                X, Y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                X, Y = next(data_iter)

            X, Y = X.to(self.device), Y.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with ctx:
                logits = self.model(X)
                
                # Reshape for CrossEntropyLoss: x (batch_size * seq_len, vocab_size) and y (batch_size * seq_len)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    Y.view(-1)
                )

            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Step the optimizer
            self.optimizer.step()

            # Logging & Checkpointing
            if step % self.config['log_every'] == 0:
                tps = self.monitor.get_tps()
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "metrics/tokens_per_sec": tps,
                    "step": step
                })
                print(f"Step {step:05d} | Loss: {loss.item():.4f} | TPS: {tps:.0f}")

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