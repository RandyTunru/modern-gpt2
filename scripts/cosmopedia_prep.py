import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def prepare_cosmopedia():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id

    # Load the dataset in streaming mode
    dataset = load_dataset(
        "HuggingFaceTB/cosmopedia", 
        "stanford",
        split="train", 
        streaming=True
    ) 
    
    # Setup binary file 
    output_filepath = "data/processed/train_data.bin"
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Open a file in append-binary mode
    with open(output_filepath, 'wb') as f:
        max_docs = dataset.num_rows if dataset.num_rows is not None else float('inf')
        
        num_tokens = 0
        for i, row in enumerate(tqdm(dataset, total=max_docs, desc="Packing Tokens")):
            if i >= max_docs:
                break
                
            text = row['text']
            
            # Tokenize the text and append EOS token
            token_ids = [bos_id] + tokenizer.encode(text, add_special_tokens=False) + [eos_id]

            # Convert to numpy array of 16-bit integers 
            token_array = np.array(token_ids, dtype=np.uint16)
            
            # Write the raw bytes directly to the disk
            f.write(token_array.tobytes())
            num_tokens += len(token_ids)

    print(f"Total tokens processed: {num_tokens}")

if __name__ == "__main__":
    print("Starting tokenization and packing pipeline...")
    prepare_cosmopedia()
    print("data/processed/train_data.bin file is ready.")