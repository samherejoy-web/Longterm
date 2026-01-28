import torch
import torch.nn as nn
from pathlib import Path
import os

# Create a minimal mock HOPE model for demo
class DemoHOPEModel(nn.Module):
    def __init__(self, vocab_size=32000, d_model=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=512, batch_first=True)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

def create_demo_checkpoint():
    print("🔨 Creating demo model checkpoint...")
    
    # Create model
    model = DemoHOPEModel(vocab_size=32000, d_model=128, n_layers=2)
    
    # Create checkpoint structure
    checkpoint = {
        'model': model.state_dict(),
        'step': 50,
        'epoch': 1,
        'config': {
            'vocab_size': 32000,
            'd_model': 128,
            'n_layers': 2,
            'block_variant': 'hope_attention'
        }
    }
    
    # Save checkpoint
    output_dir = Path('/app/artifacts/checkpoints')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / 'demo_tiny_step_050.pt'
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✅ Demo checkpoint created: {checkpoint_path}")
    print(f"   Size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return checkpoint_path

if __name__ == "__main__":
    create_demo_checkpoint()
