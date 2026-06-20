import dotenv
dotenv.load_dotenv()

import sys, pathlib

this_folder = pathlib.Path(__file__).resolve().parent
root_folder = this_folder.parent
sys.path.insert(0, str(root_folder))

import yaml
import torch
from transformers import AutoTokenizer

from src.models.gpt import GPT

def main(prompt, checkpoint_path, config_path="configs/train_gpt2_small.yaml", max_new_tokens=512, temperature=1.0, top_k=50, repetition_penalty=1.0):
    # Example configuration for GPT-2 Small

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

        model_config = {
            "vocab_size": config['vocab_size'],
            "d_model": config['d_model'],
            "num_heads": config['num_heads'],
            "d_ff": config['d_ff'],
            "num_layers": config['num_layers'],
            "max_seq_len": config['max_seq_len']
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = GPT(**model_config)

        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        input_ids = torch.tensor(tokenizer.encode(prompt, return_tensors="pt")).to(device)
        tokenized_prompt = tokenizer.tokenize(prompt)
        print(f"Input IDs: {input_ids}")
        print(f"Tokenized Prompt: {tokenized_prompt}")

        generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_token_id, repetition_penalty=repetition_penalty)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        print("Generated Text:")
        print(generated_text)

        raw_subtokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
        print("Raw Subtokens:")
        print(raw_subtokens)

        print("Length of Generated IDs:", len(generated_ids[0]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text using a trained GPT model.")
    parser.add_argument("prompt", type=str, help="The input prompt to generate text from.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--config", type=str, default="configs/train_gpt2_small.yaml", help="Path to the model configuration file.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling for generation.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty for generation.")
    args = parser.parse_args()

    main(args.prompt, args.checkpoint, args.config, args.max_new_tokens, args.temperature, args.top_k, args.repetition_penalty)