"""
Create synthetic training data for quick HOPE model training.
This generates simple patterns that the model can learn.
"""
import os
import json
from pathlib import Path

def create_synthetic_data(output_dir: str = "/app/data/synthetic", num_samples: int = 1000):
    """Create synthetic text data for training"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Pattern 1: Simple sequences
    patterns = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of artificial intelligence.",
        "HOPE architecture uses nested learning for continual adaptation.",
        "Self-modifying Titans enable in-context learning without backpropagation.",
        "The CMS module maintains hierarchical memory across multiple timescales.",
        "Fast state allows parameter updates without modifying meta-parameters.",
        "Neural networks learn patterns from data through gradient descent.",
        "Transformers use attention mechanisms to process sequential data.",
        "Language models generate text by predicting the next token.",
        "Deep learning requires large datasets and computational resources.",
    ]
    
    # Pattern 2: Question-Answer pairs
    qa_patterns = [
        ("What is HOPE?", "HOPE stands for Hierarchical Online Parameter Estimation."),
        ("What is CMS?", "CMS is the Continual Memory System with multiple hierarchical levels."),
        ("What are Titans?", "Titans are self-modifying neural network components."),
        ("What is fast state?", "Fast state enables in-context learning with parameter deltas."),
        ("How does memorization work?", "Memorization uses teach signals to update memories online."),
    ]
    
    # Pattern 3: Math sequences
    math_patterns = [
        "1 + 1 = 2",
        "2 + 2 = 4",
        "3 + 3 = 6",
        "4 + 4 = 8",
        "5 + 5 = 10",
    ]
    
    # Pattern 4: Simple narratives
    narrative_patterns = [
        "Once upon a time, there was a neural network that learned to adapt.",
        "The model processed the input and generated a coherent response.",
        "Training proceeded smoothly as the loss decreased steadily.",
        "The architecture consisted of attention layers and memory modules.",
        "Each layer transformed the hidden representations progressively.",
    ]
    
    samples = []
    
    # Generate training samples
    for i in range(num_samples):
        if i % 4 == 0:
            # Simple patterns
            text = patterns[i % len(patterns)]
        elif i % 4 == 1:
            # Q&A pairs
            q, a = qa_patterns[i % len(qa_patterns)]
            text = f"Question: {q} Answer: {a}"
        elif i % 4 == 2:
            # Math
            text = math_patterns[i % len(math_patterns)]
        else:
            # Narratives
            text = narrative_patterns[i % len(narrative_patterns)]
        
        samples.append({"text": text, "id": i})
    
    # Save as JSONL
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    # Save as plain text (alternative format)
    text_file = output_path / "train.txt"
    with open(text_file, 'w') as f:
        for sample in samples:
            f.write(sample['text'] + '\n')
    
    print(f"✅ Created {num_samples} synthetic training samples")
    print(f"📁 JSONL: {train_file}")
    print(f"📁 Text: {text_file}")
    
    # Create validation set (smaller)
    val_samples = samples[:100]
    val_file = output_path / "val.jsonl"
    with open(val_file, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"📁 Validation: {val_file} ({len(val_samples)} samples)")
    
    return str(train_file), str(val_file), str(text_file)

if __name__ == "__main__":
    train_file, val_file, text_file = create_synthetic_data()
    print("\n✨ Synthetic data generation complete!")
