#!/usr/bin/env python
"""Comprehensive multilingual LLM training CLI with checkpoint management."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import subprocess
from datetime import datetime

from colorama import Fore, Style, init
from scripts.checkpoint.manager import CheckpointManager

init(autoreset=True)


def print_banner(text: str):
    """Print a styled banner."""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}{text}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")


def start_training(args):
    """Start or resume training."""
    print_banner("\ud83d\ude80 STARTING MULTILINGUAL LLM TRAINING")
    
    # Prepare training command
    config_name = args.config or "multilingual_production"
    
    cmd = [
        "python" if args.mode == "single" else "torchrun",
    ]
    
    if args.mode == "ddp":
        cmd.extend([
            f"--nproc_per_node={args.num_gpus}",
            "train_dist.py",
        ])
    elif args.mode == "fsdp":
        cmd.extend([
            f"--nproc_per_node={args.num_gpus}",
            "train_fsdp.py",
        ])
    elif args.mode == "deepspeed":
        cmd = [
            "deepspeed",
            f"--num_gpus={args.num_gpus}",
            "train_deepspeed.py",
        ]
    else:
        cmd.append("train.py")
    
    cmd.extend([
        f"--config-name={config_name}",
    ])
    
    # Add overrides
    if args.resume:
        cmd.append(f"train.checkpoint.resume_path={args.resume}")
    
    if args.device:
        cmd.append(f"train.device={args.device}")
    
    if args.steps:
        cmd.append(f"train.steps={args.steps}")
    
    if args.batch_size:
        cmd.append(f"data.batch_size={args.batch_size}")
    
    print(f"{Fore.GREEN}Training command:{Style.RESET_ALL}")
    print(" ".join(cmd))
    print()
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print(f"\n{Fore.GREEN}\u2705 Training completed successfully!{Style.RESET_ALL}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n{Fore.RED}\u274c Training failed with error code {e.returncode}{Style.RESET_ALL}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}\u26a0\ufe0f  Training interrupted by user{Style.RESET_ALL}\n")
        sys.exit(0)


def list_checkpoints(args):
    """List available checkpoints."""
    print_banner("\ud83d\udccb AVAILABLE CHECKPOINTS")
    
    manager = CheckpointManager(args.checkpoint_dir)
    checkpoints = manager.list_checkpoints(
        run_name=args.run_name,
        tag=args.tag,
    )
    
    if not checkpoints:
        print(f"{Fore.YELLOW}No checkpoints found.{Style.RESET_ALL}\n")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s):\n")
    
    for i, cp in enumerate(checkpoints, 1):
        print(f"{i}. {Fore.CYAN}{cp['checkpoint_id']}{Style.RESET_ALL}")
        print(f"   Run: {cp['run_name']}, Step: {cp['step']}")
        print(f"   Size: {cp['size_mb']:.2f} MB")
        print(f"   Time: {cp['timestamp']}")
        if cp.get('tags'):
            print(f"   Tags: {', '.join(cp['tags'])}")
        if cp.get('metrics'):
            print(f"   Metrics: {cp['metrics']}")
        print()


def rollback_checkpoint(args):
    """Rollback to a previous checkpoint."""
    print_banner("\ud83d\udd04 CHECKPOINT ROLLBACK")
    
    manager = CheckpointManager(args.checkpoint_dir)
    
    try:
        checkpoint, metadata = manager.rollback_to(
            checkpoint_id=args.checkpoint_id,
            run_name=args.run_name,
            step=args.step,
        )
        print(f"\n{Fore.GREEN}\u2705 Successfully rolled back to:{Style.RESET_ALL}")
        print(f"   ID: {metadata['checkpoint_id']}")
        print(f"   Step: {metadata['step']}")
        print(f"   Timestamp: {metadata['timestamp']}\n")
        
        print(f"{Fore.CYAN}To resume training from this checkpoint:{Style.RESET_ALL}")
        print(f"python scripts/cli/train_multilingual.py train --resume {metadata['path']}\n")
        
    except Exception as e:
        print(f"{Fore.RED}\u274c Rollback failed: {e}{Style.RESET_ALL}\n")
        sys.exit(1)


def create_snapshot(args):
    """Create a snapshot of checkpoints."""
    print_banner("\ud83d\udcf8 CREATING CHECKPOINT SNAPSHOT")
    
    manager = CheckpointManager(args.checkpoint_dir)
    
    try:
        snapshot_name = args.snapshot_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = manager.create_snapshot(args.run_name, snapshot_name)
        
        print(f"{Fore.GREEN}\u2705 Snapshot created successfully!{Style.RESET_ALL}")
        print(f"   Location: {snapshot_dir}\n")
        
    except Exception as e:
        print(f"{Fore.RED}\u274c Snapshot creation failed: {e}{Style.RESET_ALL}\n")
        sys.exit(1)


def show_summary(args):
    """Show checkpoint summary."""
    print_banner("\ud83d\udcca CHECKPOINT SUMMARY")
    
    manager = CheckpointManager(args.checkpoint_dir)
    manager.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual LLM Training & Checkpoint Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start training
  python scripts/cli/train_multilingual.py train --mode single --device cuda:0
  
  # Start distributed training (DDP)
  python scripts/cli/train_multilingual.py train --mode ddp --num-gpus 2
  
  # Resume from checkpoint
  python scripts/cli/train_multilingual.py train --resume artifacts/checkpoints/run/step_010000.pt
  
  # List checkpoints
  python scripts/cli/train_multilingual.py list --run-name multilingual_production
  
  # Rollback to checkpoint
  python scripts/cli/train_multilingual.py rollback --checkpoint-id multilingual_production_step_005000
  
  # Create snapshot
  python scripts/cli/train_multilingual.py snapshot --run-name multilingual_production --snapshot-name backup_v1
  
  # Show summary
  python scripts/cli/train_multilingual.py summary
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Start or resume training")
    train_parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "ddp", "fsdp", "deepspeed"],
        default="single",
        help="Training mode (default: single)"
    )
    train_parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs (for distributed training)"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        help="Config name (default: multilingual_production)"
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        help="Device override (e.g., cuda:0, cpu)"
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        help="Number of training steps"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size override"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List checkpoints")
    list_parser.add_argument(
        "--run-name",
        type=str,
        help="Filter by run name"
    )
    list_parser.add_argument(
        "--tag",
        type=str,
        help="Filter by tag"
    )
    list_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Checkpoint directory"
    )
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to checkpoint")
    rollback_parser.add_argument(
        "--checkpoint-id",
        type=str,
        help="Checkpoint ID"
    )
    rollback_parser.add_argument(
        "--run-name",
        type=str,
        help="Run name"
    )
    rollback_parser.add_argument(
        "--step",
        type=int,
        help="Specific step"
    )
    rollback_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Checkpoint directory"
    )
    
    # Snapshot command
    snapshot_parser = subparsers.add_parser("snapshot", help="Create checkpoint snapshot")
    snapshot_parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Run name to snapshot"
    )
    snapshot_parser.add_argument(
        "--snapshot-name",
        type=str,
        help="Snapshot name (default: timestamp)"
    )
    snapshot_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Checkpoint directory"
    )
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show checkpoint summary")
    summary_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Checkpoint directory"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    if args.command == "train":
        start_training(args)
    elif args.command == "list":
        list_checkpoints(args)
    elif args.command == "rollback":
        rollback_checkpoint(args)
    elif args.command == "snapshot":
        create_snapshot(args)
    elif args.command == "summary":
        show_summary(args)


if __name__ == "__main__":
    main()
