#!/usr/bin/env python
"""Train multilingual SentencePiece tokenizer for Sanskrit, Hindi, English."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import tempfile
from typing import List

import sentencepiece as spm


def collect_multilingual_corpus(
    data_dir: str,
    languages: List[str],
    output_file: str,
) -> int:
    """Collect text from all languages into a single corpus file.

    Args:
        data_dir: Directory containing language-specific .jsonl files
        languages: List of languages to include
        output_file: Output corpus file path

    Returns:
        Number of lines written
    """
    data_path = Path(data_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    print("\n📦 Collecting multilingual corpus...")

    with open(output_path, "w", encoding="utf-8") as outf:
        for lang in languages:
            jsonl_file = data_path / f"{lang}_synthetic.jsonl"

            if not jsonl_file.exists():
                print(f"  ⚠️  {lang}: File not found, skipping")
                continue

            lang_lines = 0
            with open(jsonl_file, "r", encoding="utf-8") as inf:
                for line in inf:
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "").strip()
                        if text:
                            outf.write(text + "\n")
                            lang_lines += 1
                    except json.JSONDecodeError:
                        continue

            print(f"  ✅ {lang}: {lang_lines:,} samples")
            total_lines += lang_lines

    print(f"\n🎯 Total corpus size: {total_lines:,} lines\n")
    return total_lines


def train_multilingual_tokenizer(
    corpus_file: str,
    output_dir: str,
    vocab_size: int = 32000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
) -> Path:
    """Train SentencePiece tokenizer on multilingual corpus.

    Args:
        corpus_file: Path to input corpus file
        output_dir: Output directory for tokenizer model
        vocab_size: Vocabulary size
        model_type: SentencePiece model type (unigram, bpe, char, word)
        character_coverage: Character coverage (0.9995 for multilingual)

    Returns:
        Path to trained tokenizer model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_prefix = output_path / f"multilingual_spm_{vocab_size}_{model_type}"

    print("\n🛠️ Training SentencePiece tokenizer...")
    print(f"  Corpus: {corpus_file}")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Model type: {model_type}")
    print(f"  Character coverage: {character_coverage}")
    print(f"  Output: {model_prefix}.model\n")

    # Train with parameters optimized for multilingual (Devanagari + Latin)
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        # Devanagari Unicode range: U+0900-U+097F
        # Latin: U+0000-U+007F
        # We use high character_coverage to capture both scripts
        normalization_rule_name="identity",  # Preserve original text
        remove_extra_whitespaces=False,  # Keep whitespace patterns
        byte_fallback=True,  # Handle unknown characters
        split_digits=True,  # Separate digits for better numeracy
        unk_surface=" ⁇ ",  # Unknown token surface form
        num_threads=4,
    )

    model_path = Path(f"{model_prefix}.model")
    print(f"✅ Tokenizer trained successfully!")
    print(f"   Model: {model_path}")
    print(f"   Vocab: {model_prefix}.vocab\n")

    # Save tokenizer metadata
    metadata = {
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "languages": ["sanskrit", "hindi", "english"],
        "corpus_file": str(corpus_file),
        "model_path": str(model_path),
    }

    metadata_path = output_path / "tokenizer_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📝 Metadata saved: {metadata_path}\n")

    return model_path


def test_tokenizer(model_path: Path):
    """Test tokenizer on sample texts from each language."""
    print("\n🧪 Testing tokenizer...\n")

    sp = spm.SentencePieceProcessor(model_file=str(model_path))

    test_samples = {
        "Sanskrit": "विज्ञानस्य गणितस्य च विषये ज्ञानं महत्त्वपूर्णम् अस्ति।",
        "Hindi": "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग आजकल बहुत महत्वपूर्ण हैं।",
        "English": "Machine learning and artificial intelligence are transforming technology.",
        "Mixed": "AI और machine learning का भारत में विकास हो रहा है।",
    }

    for lang, text in test_samples.items():
        tokens = sp.encode(text, out_type=str)
        token_ids = sp.encode(text, out_type=int)
        decoded = sp.decode(token_ids)

        print(f"{lang}:")
        print(f"  Input:   {text}")
        print(f"  Tokens:  {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  Count:   {len(tokens)}")
        print(f"  Decoded: {decoded}")
        print()

    print(f"Vocabulary size: {sp.vocab_size():,}")
    print(f"BOS token: {sp.bos_id()} ({sp.id_to_piece(sp.bos_id())})")
    print(f"EOS token: {sp.eos_id()} ({sp.id_to_piece(sp.eos_id())})")
    print(f"UNK token: {sp.unk_id()} ({sp.id_to_piece(sp.unk_id())})")
    print(f"PAD token: {sp.pad_id()} ({sp.id_to_piece(sp.pad_id()) if sp.pad_id() >= 0 else 'N/A'})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train multilingual SentencePiece tokenizer"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic",
        help="Directory containing synthetic data (default: data/synthetic)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/tokenizer/multilingual",
        help="Output directory (default: artifacts/tokenizer/multilingual)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
        help="Model type (default: unigram)"
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage for multilingual (default: 0.9995)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🌍 MULTILINGUAL TOKENIZER TRAINING")
    print("="*70)

    # Create temporary corpus file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as tmp:
        corpus_file = tmp.name

    try:
        # Step 1: Collect corpus
        total_lines = collect_multilingual_corpus(
            data_dir=args.data_dir,
            languages=["sanskrit", "hindi", "english"],
            output_file=corpus_file,
        )

        if total_lines == 0:
            print("❌ No data found! Generate synthetic data first.")
            return

        # Step 2: Train tokenizer
        model_path = train_multilingual_tokenizer(
            corpus_file=corpus_file,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
        )

        # Step 3: Test tokenizer
        test_tokenizer(model_path)

        print("\n" + "="*70)
        print("✅ TOKENIZER TRAINING COMPLETE!")
        print("="*70)
        print(f"\nTokenizer path: {model_path}")
        print("\n👉 Next steps:")
        print("   1. Update training config with tokenizer path")
        print("   2. Process data shards with new tokenizer")
        print("   3. Start training multilingual model\n")

    finally:
        # Clean up temporary file
        import os
        if os.path.exists(corpus_file):
            os.remove(corpus_file)


if __name__ == "__main__":
    main()
