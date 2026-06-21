import os
import dotenv

dotenv.load_dotenv()

import sys, pathlib
this_folder = pathlib.Path(__file__).resolve().parent
root_folder = this_folder.parent
sys.path.insert(0, str(root_folder))

import yaml
import torch
import wandb

from src.models.gpt import GPT
from src.data.dataset import PretrainDataset, SFTDataset
from torch.utils.data import DataLoader
from src.training.trainer import Trainer
from src.utils.metrics import get_num_params

def main(config_path="configs/train_gpt2_small.yaml"):
    # 1. Load Configurations
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Setup Weights & Biases Directory
    os.makedirs(config['wandb_dir'], exist_ok=True)
    os.environ["WANDB_DIR"] = config['wandb_dir']
    
    wandb.init(
        # entity=os.getenv("WANDB_USERNAME"),
        project=config['project_name'],
        name=config['run_name'],
        config=config,
        dir=config['wandb_dir']
    )

    # 4. Initialize the Model
    model = GPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        original_seq_len=config.get('original_seq_len', config['max_seq_len'])
    )
    
    print(f"Model initialized with {get_num_params(model):.2f} M parameters.")

    start_step = 0 # Default to 0, will be updated if loading from checkpoint with is_resume=True

    if config.get('from_checkpoint'):
        print(f"Loading weights from checkpoint: {config['from_checkpoint']}")
        checkpoint = torch.load(config['from_checkpoint'], map_location='cpu')

        # Extract the state dict
        state_dict = checkpoint['model_state_dict']
        
        # Pop out any keys that are not needed or have changed in the new model architecture
        # Quickfix for RoPE and mask buffers due to sequence length changes, the model will reinitialize them based on the new max_seq_len
        state_dict.pop('freqs_cis', None)
        state_dict.pop('mask', None)
        
        model.load_state_dict(state_dict, strict=False)

        if config.get('is_resume', False):
            print("is_resume=True: Restoring optimizer and step count.")
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Get step from checkpoint, default to 0 if not found
            start_step = checkpoint.get('step', 0)
        else:
            print("is_resume=False: Starting fresh optimizer and step 0 for new phase.")

    model.to(device)

    # 5. Initialize the DataLoader
    if config.get('use_sft_masking'):
        dataset = SFTDataset(config['data_path'], config['max_seq_len'])
    else:
        dataset = PretrainDataset(config['data_path'], config['max_seq_len'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'], 
        drop_last=config['drop_last'],
        pin_memory=True
    )

    # 6. Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95) # LLaMA standard betas
    )

    if config.get('from_checkpoint') and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Quick fix to ensure learning rate and weight decay are set according to the current config, in case they differ from the checkpoint
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['learning_rate']
                param_group['weight_decay'] = config['weight_decay']
                
            print("Optimizer states successfully restored.")
        except Exception as e:
            print(f"Starting optimizer from scratch due to architectural mismatch: {e}")

    # 7. Start Training
    trainer = Trainer(model, dataloader, optimizer, config, device, start_step=start_step)
    trainer.train()
    
    # Final save
    trainer.save_checkpoint("final")
    wandb.finish()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a GPT model.")
    parser.add_argument("--config", type=str, default="configs/train_gpt2_small.yaml", help="Path to the training configuration YAML file.")

    args = parser.parse_args()

    main(args.config)