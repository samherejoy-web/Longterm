"""Enhanced checkpoint management with versioning and rollback."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import torch


class CheckpointManager:
    """Manage model checkpoints with versioning and rollback capabilities."""

    def __init__(self, checkpoint_dir: str = "artifacts/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load checkpoint registry."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"checkpoints": [], "active": None}

    def _save_registry(self):
        """Save checkpoint registry."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def _compute_checkpoint_hash(self, checkpoint_path: Path) -> str:
        """Compute SHA256 hash of checkpoint file."""
        sha256 = hashlib.sha256()
        with open(checkpoint_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def save_checkpoint(
        self,
        checkpoint: Dict,
        run_name: str,
        step: int,
        metrics: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> Path:
        """Save checkpoint with metadata.

        Args:
            checkpoint: Checkpoint dictionary (model state, optimizer, etc.)
            run_name: Name of the training run
            step: Training step number
            metrics: Optional metrics dictionary
            tags: Optional list of tags (e.g., ['best', 'milestone'])

        Returns:
            Path to saved checkpoint
        """
        # Create run directory
        run_dir = self.checkpoint_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        checkpoint_path = run_dir / f"step_{step:06d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Compute hash
        checkpoint_hash = self._compute_checkpoint_hash(checkpoint_path)

        # Create metadata
        metadata = {
            "checkpoint_id": f"{run_name}_step_{step:06d}",
            "run_name": run_name,
            "step": step,
            "path": str(checkpoint_path),
            "timestamp": datetime.now().isoformat(),
            "hash": checkpoint_hash,
            "metrics": metrics or {},
            "tags": tags or [],
            "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
        }

        # Add to registry
        self.registry["checkpoints"].append(metadata)
        self.registry["active"] = metadata["checkpoint_id"]
        self._save_registry()

        # Save separate metadata file
        metadata_path = run_dir / f"step_{step:06d}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Checkpoint saved: {checkpoint_path}")
        print(f"   Step: {step}, Size: {metadata['size_mb']:.2f} MB")
        print(f"   Hash: {checkpoint_hash[:16]}...")

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        run_name: Optional[str] = None,
        step: Optional[int] = None,
    ) -> tuple[Dict, Dict]:
        """Load checkpoint and its metadata.

        Args:
            checkpoint_id: Specific checkpoint ID
            run_name: Load latest from run
            step: Load specific step from run (requires run_name)

        Returns:
            Tuple of (checkpoint_dict, metadata_dict)
        """
        if checkpoint_id:
            # Load by ID
            metadata = next(
                (cp for cp in self.registry["checkpoints"] if cp["checkpoint_id"] == checkpoint_id),
                None
            )
        elif run_name and step is not None:
            # Load specific step from run
            checkpoint_id = f"{run_name}_step_{step:06d}"
            metadata = next(
                (cp for cp in self.registry["checkpoints"] if cp["checkpoint_id"] == checkpoint_id),
                None
            )
        elif run_name:
            # Load latest from run
            run_checkpoints = [
                cp for cp in self.registry["checkpoints"] if cp["run_name"] == run_name
            ]
            metadata = max(run_checkpoints, key=lambda x: x["step"]) if run_checkpoints else None
        else:
            # Load active checkpoint
            if self.registry["active"]:
                metadata = next(
                    (cp for cp in self.registry["checkpoints"] 
                     if cp["checkpoint_id"] == self.registry["active"]),
                    None
                )
            else:
                metadata = None

        if not metadata:
            raise ValueError("Checkpoint not found")

        checkpoint_path = Path(metadata["path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Verify hash
        current_hash = self._compute_checkpoint_hash(checkpoint_path)
        if current_hash != metadata["hash"]:
            raise ValueError(f"Checkpoint hash mismatch! File may be corrupted.")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        print(f"   Step: {metadata['step']}, Verified: ✓")

        return checkpoint, metadata

    def rollback_to(
        self,
        checkpoint_id: Optional[str] = None,
        run_name: Optional[str] = None,
        step: Optional[int] = None,
    ):
        """Rollback to a previous checkpoint (sets it as active).

        Args:
            checkpoint_id: Specific checkpoint ID
            run_name: Rollback to latest from run
            step: Rollback to specific step (requires run_name)
        """
        checkpoint, metadata = self.load_checkpoint(checkpoint_id, run_name, step)
        self.registry["active"] = metadata["checkpoint_id"]
        self._save_registry()

        print(f"🔄 Rolled back to: {metadata['checkpoint_id']}")
        print(f"   Timestamp: {metadata['timestamp']}")
        print(f"   Step: {metadata['step']}")

        return checkpoint, metadata

    def list_checkpoints(
        self,
        run_name: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Dict]:
        """List available checkpoints.

        Args:
            run_name: Filter by run name
            tag: Filter by tag

        Returns:
            List of checkpoint metadata
        """
        checkpoints = self.registry["checkpoints"]

        if run_name:
            checkpoints = [cp for cp in checkpoints if cp["run_name"] == run_name]

        if tag:
            checkpoints = [cp for cp in checkpoints if tag in cp.get("tags", [])]

        # Sort by timestamp (newest first)
        checkpoints = sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)

        return checkpoints

    def delete_checkpoint(self, checkpoint_id: str, confirm: bool = False):
        """Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete
            confirm: Must be True to actually delete
        """
        if not confirm:
            print("⚠️  Set confirm=True to delete checkpoint")
            return

        metadata = next(
            (cp for cp in self.registry["checkpoints"] if cp["checkpoint_id"] == checkpoint_id),
            None
        )

        if not metadata:
            print(f"❌ Checkpoint not found: {checkpoint_id}")
            return

        # Delete files
        checkpoint_path = Path(metadata["path"])
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

        # Remove from registry
        self.registry["checkpoints"] = [
            cp for cp in self.registry["checkpoints"] if cp["checkpoint_id"] != checkpoint_id
        ]

        # Clear active if this was active
        if self.registry["active"] == checkpoint_id:
            self.registry["active"] = None

        self._save_registry()
        print(f"✅ Deleted checkpoint: {checkpoint_id}")

    def create_snapshot(self, run_name: str, snapshot_name: str) -> Path:
        """Create a snapshot (backup) of all checkpoints in a run.

        Args:
            run_name: Name of the run
            snapshot_name: Name for the snapshot

        Returns:
            Path to snapshot directory
        """
        run_dir = self.checkpoint_dir / run_name
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")

        snapshot_dir = self.checkpoint_dir / "snapshots" / f"{run_name}_{snapshot_name}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files
        shutil.copytree(run_dir, snapshot_dir, dirs_exist_ok=True)

        # Save snapshot metadata
        snapshot_metadata = {
            "run_name": run_name,
            "snapshot_name": snapshot_name,
            "timestamp": datetime.now().isoformat(),
            "path": str(snapshot_dir),
        }

        metadata_path = snapshot_dir / "snapshot_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(snapshot_metadata, f, indent=2)

        print(f"📸 Snapshot created: {snapshot_dir}")
        return snapshot_dir

    def print_summary(self):
        """Print checkpoint summary."""
        print("\n" + "="*70)
        print("CHECKPOINT REGISTRY SUMMARY")
        print("="*70 + "\n")

        print(f"Total checkpoints: {len(self.registry['checkpoints'])}")
        print(f"Active checkpoint: {self.registry.get('active', 'None')}\n")

        # Group by run
        runs = {}
        for cp in self.registry["checkpoints"]:
            run = cp["run_name"]
            if run not in runs:
                runs[run] = []
            runs[run].append(cp)

        print("Checkpoints by run:")
        for run, checkpoints in runs.items():
            total_size = sum(cp["size_mb"] for cp in checkpoints)
            steps = [cp["step"] for cp in checkpoints]
            print(f"  {run}:")
            print(f"    - Count: {len(checkpoints)}")
            print(f"    - Steps: {min(steps)} to {max(steps)}")
            print(f"    - Total size: {total_size:.2f} MB")

        print("\n" + "="*70 + "\n")
