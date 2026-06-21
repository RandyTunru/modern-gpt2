import dotenv

dotenv.load_dotenv()

import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def prepare_openhermes(split, output_file, max_docs=None, min_tokens=10, max_seq_len=2048, chunk_size=10000, min_turns=1):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    pad_id = eos_id # Mistral doesn't have a pad token, so we use EOS as a placeholder for padding.

    dataset = load_dataset(
        "HuggingFaceTB/OpenHermes-2.5-H4",
        split=split, 
        streaming=True
    ) 
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    f_tokens = open(output_file, 'wb')
    f_labels = open(output_file.replace('.bin', '_labels.bin'), 'wb')
    f_doc_ids = open(output_file.replace('.bin', '_doc_ids.bin'), 'wb')
    
    document_buffer = []
    total_bins_written = 0
    doc_count = 0

    def process_buffer(buffer):
        nonlocal total_bins_written
        buffer.sort(key=lambda x: len(x[0]), reverse=True)
        
        bins_tokens = []
        bins_labels = [] # Store the labels corresponding to each token in the bins. User tokens get -100, assistant tokens get their own token IDs, and padding gets -100.
        bins_doc_ids = [] # Track the document IDs for each bin
        
        for doc_tokens, doc_labels in buffer:
            placed = False
            
            # Try to fit the document into an existing bin
            for i in range(len(bins_tokens)):
                if len(bins_tokens[i]) + len(doc_tokens) <= max_seq_len:
                    # The new doc gets an ID based on how many docs are already in this bin
                    # If there's 1 doc, new ID is 2.
                    current_doc_id = len(set(bins_doc_ids[i])) + 1 
                    
                    bins_tokens[i].extend(doc_tokens)
                    bins_labels[i].extend(doc_labels)
                    bins_doc_ids[i].extend([current_doc_id] * len(doc_tokens))
                    
                    placed = True
                    break
            
            # If it doesn't fit anywhere, create a new bin
            if not placed: 
                if len(doc_tokens) <= max_seq_len:
                    bins_tokens.append(list(doc_tokens))
                    bins_labels.append(list(doc_labels))
                    bins_doc_ids.append([1] * len(doc_tokens)) # First document in new bin gets ID 1
                    
        # Pad the bins and write to disk
        for b_tok, b_lab, b_ids in zip(bins_tokens, bins_labels, bins_doc_ids):
            pad_len = max_seq_len - len(b_tok)
            
            b_tok.extend([pad_id] * pad_len)
            b_lab.extend([-100] * pad_len) 
            b_ids.extend([0] * pad_len) # 0 means padding
            
            # Write all three arrays
            f_tokens.write(np.array(b_tok, dtype=np.uint16).tobytes())
            f_labels.write(np.array(b_lab, dtype=np.int32).tobytes()) # Using int32 to prevent overflow
            f_doc_ids.write(np.array(b_ids, dtype=np.int16).tobytes()) # int16 is plenty for doc IDs
            
            total_bins_written += 1
    
    for i, row in enumerate(tqdm(dataset, total=max_docs, desc="Reading streaming dataset")):
        if max_docs and i >= max_docs:
            break
            
        conversations = row['messages']

        if len(conversations) < min_turns * 2:  # Each turn consists of a user and an assistant message
            continue

        doc_tokens = [bos_id]
        doc_labels = [-100] # Mask the BOS token

        for turn in conversations:
            role = turn['role']
            content = turn['content']
            
            if role == 'user':
                # Mistral Format: [INST] {prompt} [/INST]
                text = f"[INST] {content} [/INST]"
                tokens = tokenizer.encode(text, add_special_tokens=False)
                doc_tokens.extend(tokens)
                doc_labels.extend([-100] * len(tokens)) # MASK USER
            else:
                # Assistant response, appended with EOS so it learns to stop
                text = f" {content}"
                tokens = tokenizer.encode(text, add_special_tokens=False) + [eos_id]
                doc_tokens.extend(tokens)
                doc_labels.extend(tokens) 
        
        if min_tokens <= len(doc_tokens) <= max_seq_len:
            document_buffer.append((doc_tokens, doc_labels))
            
        doc_count += 1
        
        # When buffer is full, process it and clear memory
        if len(document_buffer) >= chunk_size:
            process_buffer(document_buffer)
            document_buffer = []

    # Process any remaining documents
    if document_buffer:
        process_buffer(document_buffer)

    f_tokens.close()
    f_labels.close()
    f_doc_ids.close()

    # Write a sidecar metadata file so the Dataset can verify it's being read at the
    # exact sequence length it was packed for, and load each array with the right dtype.
    meta = {
        "max_seq_len": max_seq_len,
        "num_bins": total_bins_written,
        "token_dtype": "uint16",
        "label_dtype": "int32",
        "doc_id_dtype": "int16",
    }
    with open(output_file.replace('.bin', '_meta.json'), 'w') as f_meta:
        json.dump(meta, f_meta, indent=2)

    print(f"Total bins written: {total_bins_written}")
    print(f"Total documents processed: {doc_count}")
    print("Total tokens written to file:", total_bins_written * max_seq_len)

    os._exit(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare the Everyday Conversations dataset for training.")
    parser.add_argument("--split", type=str, default="train_sft", help="Dataset split to process (e.g., train, validation)")
    parser.add_argument("--output-file", type=str, default="data/processed/train_data.bin", help="Output file path")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of documents to process (for testing)")
    parser.add_argument("--min-tokens", type=int, default=512, help="Minimum number of tokens per document to be included in the output")
    parser.add_argument("--max-seq-len", type=int, required=True, help="Maximum sequence length of each document")
    parser.add_argument("--min-turns", type=int, default=1, help="Minimum number of conversation turns per document to be included in the output (a turn consists of a user and an assistant message)")

    args = parser.parse_args()

    if args.max_seq_len is not None and args.max_seq_len < args.min_tokens:
        parser.error("--max-seq-len must be greater than or equal to --min-tokens")
    if args.min_turns < 1:
        parser.error("--min-turns must be at least 1")

    print("Starting tokenization and packing pipeline...")
    prepare_openhermes(
        split=args.split,
        output_file=args.output_file,
        max_docs=args.max_docs,
        min_tokens=args.min_tokens,
        max_seq_len=args.max_seq_len,
        min_turns=args.min_turns
    )
    print(f"{args.output_file} file is ready.")