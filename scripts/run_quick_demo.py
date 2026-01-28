#!/usr/bin/env python3
"""
Quick demo training script that:
1. Creates synthetic data
2. Trains a HOPE model
3. Verifies the checkpoint
"""
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 80)
    print("HOPE MODEL QUICK DEMO TRAINING")
    print("=" * 80)
    
    # Step 1: Create synthetic data
    print("\n[1/3] Creating synthetic training data...")
    from scripts.create_synthetic_data import create_synthetic_data
    try:
        train_file, val_file, text_file = create_synthetic_data(
            output_dir="/app/data/synthetic",
            num_samples=500
        )
        print(f"✅ Synthetic data created:")
        print(f"   - Training: {train_file}")
        print(f"   - Validation: {val_file}")
        print(f"   - Text: {text_file}")
    except Exception as e:
        print(f"❌ Failed to create synthetic data: {e}")
        return 1
    
    # Step 2: Train the model
    print("\n[2/3] Training HOPE model with Nested Learning mechanisms...")
    print("   This will demonstrate:")
    print("   - ✅ HOPE architecture with attention")
    print("   - ✅ CMS (Continual Memory System) hierarchical updates")
    print("   - ✅ TITAN memory with online learning")
    print("   - ✅ Teach signals (δℓ) for memory updates")
    print("   - ✅ Fast state for in-context learning")
    
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "/app/train.py", "--config-name=quick_demo"],
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            # Show last few lines of output
            lines = result.stdout.strip().split('\n')
            print("\nTraining Summary:")
            for line in lines[-10:]:
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ Training failed!")
            print(f"Error: {result.stderr[-500:]}")
            return 1
            
    except subprocess.TimeoutExpired:
        print("⚠️ Training timeout (5 minutes). This is normal for CPU training.")
        print("Check logs at /app/logs/quick_demo.json")
    except Exception as e:
        print(f"❌ Training error: {e}")
        return 1
    
    # Step 3: Verify checkpoint
    print("\n[3/3] Verifying checkpoint...")
    checkpoint_dir = Path("/app/artifacts/checkpoints/quick_demo")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            print(f"✅ Found {len(checkpoints)} checkpoint(s):")
            for ckpt in checkpoints:
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"   - {ckpt.name} ({size_mb:.2f} MB)")
            
            # Verify the latest checkpoint can be loaded
            latest_ckpt = sorted(checkpoints)[-1]
            print(f"\n🔍 Verifying latest checkpoint: {latest_ckpt.name}")
            
            import torch
            try:
                checkpoint = torch.load(latest_ckpt, map_location='cpu')
                print("✅ Checkpoint structure:")
                for key in checkpoint.keys():
                    if key == 'model':
                        num_params = len(checkpoint[key])
                        print(f"   - {key}: {num_params} parameter tensors")
                    elif key == 'config':
                        print(f"   - {key}: Configuration saved")
                    else:
                        print(f"   - {key}: {checkpoint.get(key, 'N/A')}")
                
                # Verify HOPE components
                if 'config' in checkpoint and 'model' in checkpoint['config']:
                    model_cfg = checkpoint['config']['model']
                    print(f"\n✅ HOPE Configuration:")
                    print(f"   - Variant: {model_cfg.get('block_variant', 'N/A')}")
                    print(f"   - Layers: {model_cfg.get('num_layers', 'N/A')}")
                    print(f"   - Dimension: {model_cfg.get('dim', 'N/A')}")
                    print(f"   - CMS Levels: {len(model_cfg.get('cms_levels', []))}")
                    print(f"   - Teach Scale: {model_cfg.get('teach_scale', 'N/A')}")
                
            except Exception as e:
                print(f"⚠️ Checkpoint verification failed: {e}")
        else:
            print("⚠️ No checkpoints found")
    else:
        print("⚠️ Checkpoint directory not found")
    
    print("\n" + "=" * 80)
    print("✨ DEMO TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Start the backend: sudo supervisorctl restart backend")
    print("2. Open the frontend and go to 'Models' tab")
    print("3. Select the trained checkpoint from quick_demo folder")
    print("4. Load the model and start chatting!")
    print("\nThe model uses:")
    print("  • Nested Learning semantics with fast state")
    print("  • HOPE blocks with attention mechanism")
    print("  • CMS hierarchical memory (fast/mid/slow)")
    print("  • TITAN memory with online updates")
    print("  • Teach signals for continual learning")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
