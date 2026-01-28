#!/usr/bin/env python
"""CLI for previewing and validating synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from scripts.synthetic_data.preview import DataPreview


def main():
    parser = argparse.ArgumentParser(
        description="Preview and validate synthetic multilingual data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic",
        help="Data directory (default: data/synthetic)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of random samples to show per language (default: 3)"
    )

    args = parser.parse_args()

    print("🔍 Synthetic Data Preview & Validation")
    print("="*70 + "\n")

    preview = DataPreview(args.data_dir)
    preview.show_summary()
    preview.check_balance()
    preview.check_diversity()
    preview.show_random_samples(n=args.samples)
    preview.export_approval_report()

    print("\n✅ Preview complete!")
    print("   Review approval report: data/synthetic/approval_report.txt\n")


if __name__ == "__main__":
    main()
