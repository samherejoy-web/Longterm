"""
Quick training script for creating a demo HOPE model with synthetic data.
This creates a small, fast-training model for UI demonstration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os

from src.nested_learning.model import HOPEModel, ModelConfig
from src.nested_learning.levels import LevelSpec
from src.nested_learning.tokenizer import SentencePieceTokenizer

class SimpleTextDataset(Dataset):
    """Simple dataset for text training"""
    def __init__(self, data_file: str, tokenizer=None, max_length: int = 64):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r') as f:
            if data_file.endswith('.jsonl'):
                for line in f:
                    data = json.loads(line)
                    self.samples.append(data['text'])
            else:
                self.samples = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            # Pad or truncate
            if len(tokens) < self.max_length:
                tokens = tokens + [0] * (self.max_length - len(tokens))
            else:
                tokens = tokens[:self.max_length]
            return torch.tensor(tokens, dtype=torch.long)
        else:
            # Simple character-level tokenization for demo
            # Map characters to integers
            tokens = [ord(c) % 256 for c in text[:self.max_length]]
            if len(tokens) < self.max_length:
                tokens = tokens + [0] * (self.max_length - len(tokens))
            return torch.tensor(tokens, dtype=torch.long)

def create_demo_model(vocab_size: int = 256, device: str = "cpu"):
    """Create a small HOPE model for demo purposes"""
    config = ModelConfig(
        vocab_size=vocab_size,
        dim=128,  # Small dimension for fast training
        num_layers=4,  # Few layers
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=4),
        cms_levels=[
            LevelSpec(name="fast", update_period=2),
            LevelSpec(name="slow", update_period=8),
        ],
        cms_flush_partial_at_end=True,
        cms_use_layernorm=True,
        teach_scale=1.0,
        teach_clip=0.0,
        gradient_checkpointing=False,
        qk_l2_norm=True,
        self_mod_lr=1e-3,
        block_variant="hope_selfmod",  # Use self-modifying variant
        self_mod_chunk_size=1,
        self_mod_objective="l2",
        self_mod_stopgrad_vhat=True,
        self_mod_use_rank1_precond=True,
        self_mod_use_alpha=True,
        self_mod_adaptive_q=False,
        self_mod_local_conv_window=4,
    )
    
    model = HOPEModel(config).to(device)
    return model, config

def train_demo_model(
    data_file: str,
    output_dir: str = "/app/artifacts/demo",
    steps: int = 500,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    save_every: int = 100,
):
    """Train a demo model on synthetic data"""
    
    print(f"🚀 Starting demo training")
    print(f"📁 Data: {data_file}")
    print(f"💾 Output: {output_dir}")
    print(f"🔢 Steps: {steps}")
    print(f"🎯 Device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = SimpleTextDataset(data_file, tokenizer=None, max_length=32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model, config = create_demo_model(vocab_size=256, device=device)
    
    # Initialize fast state for nested learning
    fast_state = model.init_fast_state()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    step = 0
    epoch = 0
    total_loss = 0
    
    progress = tqdm(total=steps, desc="Training")
    
    while step < steps:
        epoch += 1
        for batch in dataloader:
            if step >= steps:
                break
            
            tokens = batch.to(device)
            
            # Forward pass
            # Input: all but last token, Target: all but first token
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            
            logits = model(input_tokens)
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
                ignore_index=0  # Ignore padding
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            # Update progress
            progress.update(1)
            progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/step:.4f}',
                'epoch': epoch
            })
            
            # Save checkpoint
            if step % save_every == 0 or step == steps:
                checkpoint_path = os.path.join(output_dir, f"demo_step_{step:06d}.pt")
                torch.save({
                    'step': step,
                    'model': model.state_dict(),
                    'config': config.__dict__,
                    'optimizer': optimizer.state_dict(),
                    'loss': total_loss / step,
                }, checkpoint_path)
                print(f"\n💾 Saved checkpoint: {checkpoint_path}")
    
    progress.close()
    
    # Save final model
    final_path = os.path.join(output_dir, "demo_final.pt")
    torch.save({
        'step': steps,
        'model': model.state_dict(),
        'config': config.__dict__,
        'optimizer': optimizer.state_dict(),
        'loss': total_loss / step,
    }, final_path)
    
    # Save config separately
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    
    print(f"\n✅ Training complete!")
    print(f"📊 Final loss: {total_loss/steps:.4f}")
    print(f"💾 Final checkpoint: {final_path}")
    print(f"⚙️ Config: {config_path}")
    
    return final_path, config_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick train demo HOPE model")
    parser.add_argument("--data", type=str, default="/app/data/synthetic/train.txt",
                        help="Path to training data")
    parser.add_argument("--output", type=str, default="/app/artifacts/demo",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=500,
                        help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Check if data exists, create if not
    if not os.path.exists(args.data):
        print(f"⚠️ Data file not found: {args.data}")
        print("📝 Creating synthetic data...")
        from create_synthetic_data import create_synthetic_data
        train_file, _, text_file = create_synthetic_data()
        args.data = text_file
    
    train_demo_model(
        data_file=args.data,
        output_dir=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
