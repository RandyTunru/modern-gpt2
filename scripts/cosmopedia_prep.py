import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def prepare_cosmopedia(subset, split, output_file, max_docs=None):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id

    # Load the dataset in streaming mode
    dataset = load_dataset(
        "HuggingFaceTB/cosmopedia", 
        subset,
        split=split, 
        streaming=True
    ) 
    
    # Setup binary file 
    output_filepath = output_file
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Open a file in append-binary mode
    with open(output_filepath, 'wb') as f:
        if max_docs is None:
            max_docs = float('inf')
        
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
    import argparse
    parser = argparse.ArgumentParser(description="Prepare the Cosmopedia dataset for training.")
    parser.add_argument("--subset", type=str, default="stanford", help="Dataset subset to process (look for available subsets in HuggingFaceTB/cosmopedia)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process (e.g., train, validation)")
    parser.add_argument("--output-file", type=str, default="data/processed/train_data.bin", help="Output file path")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of documents to process (for testing)")

    args = parser.parse_args()

    print("Starting tokenization and packing pipeline...")
    prepare_cosmopedia(
        subset=args.subset,
        split=args.split,
        output_file=args.output_file,
        max_docs=args.max_docs
    )
    print(f"{args.output_file} file is ready.")