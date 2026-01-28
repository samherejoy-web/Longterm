#!/usr/bin/env python3
"""
Simple training script that can be run from the backend API
"""
import sys
import os
import json
import subprocess
from pathlib import Path

def run_training(config_name="pilot_smoke", steps=100, device="cpu", checkpoint_dir=None):
    """
    Run training with specified configuration
    
    Args:
        config_name: Name of the config file (without .yaml)
        steps: Number of training steps
        device: Device to use (cpu/cuda/mps)
        checkpoint_dir: Optional checkpoint directory override
    """
    # Ensure we're in the app directory
    os.chdir('/app')
    
    # Build command
    cmd = [
        sys.executable,
        "train.py",
        f"--config-name={config_name}",
        f"train.steps={steps}",
        f"train.device={device}",
    ]
    
    if checkpoint_dir:
        cmd.append(f"train.checkpoint.dir={checkpoint_dir}")
    
    # Log the command
    print(f"Running training command: {' '.join(cmd)}")
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd="/app"
        )
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": ' '.join(cmd)
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "command": ' '.join(cmd)
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple training runner")
    parser.add_argument("--config-name", default="pilot_smoke", help="Config name")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory")
    parser.add_argument("--output-json", default=None, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    result = run_training(
        config_name=args.config_name,
        steps=args.steps,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Print results
    print("\n" + "="*80)
    print("TRAINING RESULTS")
    print("="*80)
    print(f"Success: {result['success']}")
    print(f"Return Code: {result['returncode']}")
    print(f"Command: {result['command']}")
    print("\nSTDOUT:")
    print(result['stdout'][-1000:] if len(result['stdout']) > 1000 else result['stdout'])
    print("\nSTDERR:")
    print(result['stderr'][-1000:] if len(result['stderr']) > 1000 else result['stderr'])
    
    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")
    
    sys.exit(0 if result['success'] else 1)
