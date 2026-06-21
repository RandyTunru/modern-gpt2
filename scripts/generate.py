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

def format_user_turn(content):
    # Must mirror the packing in data prep (eg. check scripts/openhermes_prep.py) exactly: "[INST] {content} [/INST]".
    return f"[INST] {content} [/INST]"


def run_multiturn(model, tokenizer, device, gen_kwargs, first_prompt=None):
    conversation_ids = [tokenizer.bos_token_id]
    print("Multi-turn chat. Type 'exit' or 'quit' to stop.\n")

    while True:
        user = first_prompt if first_prompt is not None else input("User: ")
        first_prompt = None
        if user.strip().lower() in {"exit", "quit"}:
            break

        conversation_ids += tokenizer.encode(format_user_turn(user), add_special_tokens=False)

        input_ids = torch.tensor([conversation_ids], device=device)
        generated_ids = model.generate(input_ids, eos_id=tokenizer.eos_token_id, **gen_kwargs)

        # Everything past the prompt is the new assistant turn (includes trailing EOS if it stopped).
        new_tokens = generated_ids[0][len(conversation_ids):].tolist()
        conversation_ids += new_tokens

        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"Assistant: {response}\n")


def main(prompt, checkpoint_path, config_path="configs/train_gpt2_small.yaml", max_new_tokens=512, temperature=1.0, top_k=50, repetition_penalty=1.0, apply_chat_template=False, multiturn=False):
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

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        if multiturn:
            run_multiturn(model, tokenizer, device, gen_kwargs, first_prompt=prompt)
            return

        if apply_chat_template:
            prompt = format_user_turn(prompt)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        tokenized_prompt = tokenizer.tokenize(prompt)
        print(f"Input IDs: {input_ids}")
        print(f"Tokenized Prompt: {tokenized_prompt}")

        generated_ids = model.generate(input_ids, eos_id=tokenizer.eos_token_id, **gen_kwargs)

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
    parser.add_argument("--apply-chat-template", action="store_true", help="Use the chat template for the prompt.")
    parser.add_argument("--multiturn", action="store_true", help="Interactive multi-turn chat (uses the chat template; the positional prompt becomes the first user message).")
    args = parser.parse_args()

    main(args.prompt, args.checkpoint, args.config, args.max_new_tokens, args.temperature, args.top_k, args.repetition_penalty, args.apply_chat_template, args.multiturn)