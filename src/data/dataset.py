import json
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
        self.num_samples = self.total_tokens // self.seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # We grab seq_length tokens from the disk because we need to create input-target pairs where the target is the next token after the input sequence.
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length # No shift, we will handle in the train loop
        
        chunk = self.data[start_idx : end_idx]
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))
        
        return {
            "input_ids": chunk_tensor,
            "labels": chunk_tensor.clone()  # Labels are are a copy of the input_ids, the shift will be handled in the training loop to align predictions with targets
        }
     
class SFTDataset(Dataset):
    def __init__(self, data_path, seq_length):
        super().__init__()
        self.seq_length = seq_length

        # Load packing metadata to verify correct sequence length and data types
        with open(data_path.replace('.bin', '_meta.json')) as f_meta:
            meta = json.load(f_meta)

        if seq_length != meta['max_seq_len']:
            raise ValueError(
                f"seq_length ({seq_length}) must equal the packed max_seq_len "
                f"({meta['max_seq_len']}). This packed data can only be trained at that length, "
                f"because doc_ids are bin-relative and a mismatched stride would straddle bins."
            )

        self.data = np.memmap(data_path, dtype=np.dtype(meta['token_dtype']), mode='r')
        self.labels = np.memmap(data_path.replace('.bin', '_labels.bin'), dtype=np.dtype(meta['label_dtype']), mode='r')
        self.doc_ids = np.memmap(data_path.replace('.bin', '_doc_ids.bin'), dtype=np.dtype(meta['doc_id_dtype']), mode='r')

        self.total_tokens = len(self.data)
        self.num_samples = self.total_tokens // self.seq_length

        if self.total_tokens % seq_length != 0:
            raise ValueError(
                f"Token count ({self.total_tokens}) is not divisible by seq_length ({seq_length}); "
                f"the data file looks truncated or corrupt."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        
        chunk = self.data[start_idx : end_idx]
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))

        label_chunk = self.labels[start_idx : end_idx]
        label_tensor = torch.from_numpy(label_chunk.astype(np.int64))

        doc_id_chunk = self.doc_ids[start_idx : end_idx]
        doc_id_tensor = torch.from_numpy(doc_id_chunk.astype(np.int64))
        
        return {
            "input_ids": chunk_tensor,
            "labels": label_tensor,
            "doc_ids": doc_id_tensor
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the PretrainDataset")
    parser.add_argument('--data-path', type=str, default='data/processed/train_data.bin', help='Path to the binary data file')
    parser.add_argument('--seq-length', type=int, default=1024, help='Sequence length for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for DataLoader')

    args = parser.parse_args()

    train_dataset = PretrainDataset(args.data_path, args.seq_length)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    print("DataLoader length: ", len(train_dataloader))
    
    # Just check the first batch for demonstration purposes
    for X_batch, Y_batch in train_dataloader:
        print("Input batch shape:", X_batch.shape)  # Should be (batch_size, seq_length)
        print("Target batch shape:", Y_batch.shape)  # Should be (batch_size, seq_length)
        print("Input matches target (shifted): ", not any(X_batch[0][1:] != Y_batch[0][:-1]))  # Should be True for all positions since Y is just X shifted by one token
        break  