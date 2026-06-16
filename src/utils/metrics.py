import time
import torch
import yaml

def get_num_params(model):
    """Returns the total number of trainable parameters in millions."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params / 1e6

class ThroughputMonitor:
    def __init__(self, batch_size, seq_len):
        self.tokens_per_batch = batch_size * seq_len
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def get_tps(self):
        """Calculates Tokens Per Second (TPS)"""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        tps = self.tokens_per_batch / elapsed
        self.start_time = time.time() # Reset for next measurement
        return tps
    
if __name__ == "__main__":
    import sys, pathlib

    this_folder = pathlib.Path(__file__).resolve().parent
    root_folder = this_folder.parent.parent
    sys.path.insert(0, str(root_folder))

    from src.models.gpt import GPT

    with open("configs/train_gpt2_small.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = GPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len']
    )

    print(f"Model has {get_num_params(model):.2f} M trainable parameters.")