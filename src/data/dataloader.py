import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class PretrainDataset(Dataset):
    def __init__(self, data_path, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        self.total_tokens = len(self.data)
        self.num_samples = (self.total_tokens - 1) // self.seq_length # Minus 1 because we need seq_length + 1 tokens for input-target pairs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # We grab seq_length + 1 tokens from the disk because we need to create input-target pairs where the target is the next token after the input sequence.
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        
        chunk = self.data[start_idx : end_idx]
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))
        
        # X gets the first seq_length tokens, Y gets shifted by one token to the right (the next token for each position in X)
        X = chunk_tensor[:-1]
        Y = chunk_tensor[1:]
        
        return X, Y

seq_length = 4096
batch_size = 32

train_dataset = PretrainDataset("data/processed/train_data.bin", seq_length)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)

if __name__ == "__main__":
    # Just check the first batch for demonstration purposes
    for X_batch, Y_batch in train_dataloader:
        print("Input batch shape:", X_batch.shape)  # Should be (batch_size, seq_length)
        print("Target batch shape:", Y_batch.shape)  # Should be (batch_size, seq_length)
        print("Input matches target (shifted): ", not any(X_batch[0][1:] != Y_batch[0][:-1]))  # Should be True for all positions since Y is just X shifted by one token
        break  