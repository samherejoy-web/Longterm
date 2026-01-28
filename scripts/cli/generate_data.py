#!/usr/bin/env python
"""CLI for generating synthetic multilingual data."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from scripts.synthetic_data.generator import SyntheticDataGenerator
from scripts.synthetic_data.preview import DataPreview


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic multilingual data for LLM training"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Groq API key"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Total number of samples to generate (default: 10000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory (default: data/synthetic)"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip LLM refinement (faster, template-only)"
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Generate small preview dataset (100 samples)"
    )
    parser.add_argument(
        "--with-reasoning",
        action="store_true",
        help="Also generate reasoning chains"
    )
    parser.add_argument(
        "--reasoning-count",
        type=int,
        default=1000,
        help="Number of reasoning chains to generate (default: 1000)"
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show data preview after generation"
    )

    args = parser.parse_args()

    print("🚀 Synthetic Multilingual Data Generator")
    print("="*70 + "\n")

    # Initialize generator
    generator = SyntheticDataGenerator(
        groq_api_key=args.api_key,
        output_dir=args.output_dir,
    )

    # Generate main dataset
    print("🎯 Step 1: Generating main dataset...")
    data = generator.generate_balanced_dataset(
        total_samples=args.samples,
        refine=not args.no_refine,
        preview_mode=args.preview_only,
    )

    # Save dataset
    print("\n🎯 Step 2: Saving dataset...")
    output_dir = generator.save_dataset(data, format="jsonl")

    # Generate reasoning chains if requested
    if args.with_reasoning:
        print("\n🎯 Step 3: Generating reasoning chains...")
        generator.generate_reasoning_chains(count=args.reasoning_count)

    print(f"\n✅ Data generation complete!")
    print(f"   Output directory: {output_dir}")

    # Show preview if requested
    if args.show_preview:
        print("\n🎯 Step 4: Generating preview...")
        preview = DataPreview(str(output_dir))
        preview.run_full_preview()

    print("\n👉 Next steps:")
    print("   1. Review the approval report: data/synthetic/approval_report.txt")
    print("   2. Run preview tool: python scripts/cli/preview_data.py")
    print("   3. Train tokenizer: python scripts/data/train_tokenizer.py")
    print("   4. Start training: python scripts/cli/train_multilingual.py\n")


if __name__ == "__main__":
    main()
