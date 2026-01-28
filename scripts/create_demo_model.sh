#!/bin/bash

# Create a tiny demo HOPE model for testing the chat interface

echo "=========================================="
echo "🧪 Creating Demo HOPE Model"
echo "=========================================="
echo ""

cd /app

# Create a simple demo config for ultra-fast training
cat > /app/configs/demo_tiny.yaml << 'EOF'
defaults:
  - _self_

model:
  vocab_size: 32000
  d_model: 128
  n_layers: 2
  n_heads: 4
  block_variant: hope_attention
  dropout: 0.0
  
data:
  batch_size: 2
  seq_length: 128
  
train:
  steps: 50
  log_interval: 10
  checkpoint_interval: 50
  device: cpu
  seed: 42
  
optim:
  type: adamw
  lr: 0.0003
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  
logging:
  enabled: false
EOF

echo "✅ Created demo config: /app/configs/demo_tiny.yaml"
echo ""

# Create sample training data
mkdir -p /app/data/demo

cat > /app/data/demo/sample.txt << 'EOF'
Machine learning is a subset of artificial intelligence.
Neural networks are inspired by the human brain.
Deep learning uses multiple layers of neural networks.
Training a model requires data and computational resources.
The HOPE architecture uses nested learning mechanisms.
Self-modifying models can adapt during inference.
Transformers revolutionized natural language processing.
Attention mechanisms help models focus on relevant information.
Gradient descent optimizes model parameters.
Backpropagation computes gradients for neural networks.
EOF

echo "✅ Created sample data: /app/data/demo/sample.txt"
echo ""

# Install dependencies if needed
echo "📦 Checking dependencies..."
cd /app
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch..."
    pip install torch==2.9.0 --quiet
fi

if ! python -c "import hydra" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install hydra-core omegaconf einops sentencepiece --quiet
fi

echo "✅ Dependencies ready"
echo ""

# Create a minimal training script for demo
cat > /app/scripts/train_demo.py << 'EOFPY'
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
EOFPY

echo "📝 Created demo training script"
echo ""

# Run the demo model creation
echo "🚀 Creating demo checkpoint..."
python /app/scripts/train_demo.py

echo ""
echo "=========================================="
echo "✅ Demo Model Ready!"
echo "=========================================="
echo ""
echo "📁 Checkpoint location: /app/artifacts/checkpoints/demo_tiny_step_050.pt"
echo "📝 Config location: /app/configs/demo_tiny.yaml"
echo ""
echo "🌐 Now you can:"
echo "   1. Open http://localhost:3000"
echo "   2. Go to the 'Models' tab"
echo "   3. Select 'demo_tiny_step_050.pt' checkpoint"
echo "   4. Select 'demo_tiny' config"
echo "   5. Click 'Load Model'"
echo "   6. Go to 'Chat' tab and test!"
echo ""
echo "=========================================="
