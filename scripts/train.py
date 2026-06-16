import os
import yaml
import torch
import wandb

import pathlib
this_folder = pathlib.Path(__file__).resolve().parent
root_folder = this_folder.parent.parent
os.insert(0, str(root_folder))

from src.models.gpt import GPT
from src.data.dataloader import PretrainDataset
from torch.utils.data import DataLoader
from src.training.trainer import Trainer
from src.utils.metrics import get_num_params

def main():
    # 1. Load Configurations
    with open("configs/train_gpt2_small.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Setup Weights & Biases Directory
    os.makedirs(config['wandb_dir'], exist_ok=True)
    os.environ["WANDB_DIR"] = config['wandb_dir']
    
    wandb.init(
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
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    print(f"Model initialized with {get_num_params(model):.2f} M parameters.")

    # 5. Initialize the DataLoader
    dataset = PretrainDataset(config['data_path'], config['max_seq_len'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # 6. Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95) # LLaMA standard betas
    )

    # 7. Start Training
    trainer = Trainer(model, dataloader, optimizer, config, device)
    trainer.train()
    
    # Final save
    trainer.save_checkpoint("final")
    wandb.finish()

if __name__ == "__main__":
    main()