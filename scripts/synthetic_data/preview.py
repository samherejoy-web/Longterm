"""Interactive data preview and validation tool."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)


class DataPreview:
    """Preview and validate synthetic data before training."""

    def __init__(self, data_dir: str = "data/synthetic"):
        self.data_dir = Path(data_dir)
        self.data: Dict[str, List[str]] = {}
        self.load_data()

    def load_data(self):
        """Load generated synthetic data."""
        for lang in ["sanskrit", "hindi", "english"]:
            jsonl_file = self.data_dir / f"{lang}_synthetic.jsonl"
            if jsonl_file.exists():
                self.data[lang] = []
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        self.data[lang].append(obj["text"])
                print(f"{Fore.GREEN}✓{Style.RESET_ALL} Loaded {len(self.data[lang])} {lang} samples")
            else:
                print(f"{Fore.YELLOW}!{Style.RESET_ALL} No data found for {lang}")

    def show_summary(self):
        """Display dataset summary statistics."""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}DATASET SUMMARY")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")

        table_data = []
        total_samples = 0
        total_chars = 0

        for lang, texts in self.data.items():
            num_samples = len(texts)
            total_samples += num_samples

            if texts:
                avg_length = sum(len(t) for t in texts) / num_samples
                total_chars += sum(len(t) for t in texts)
                min_length = min(len(t) for t in texts)
                max_length = max(len(t) for t in texts)
            else:
                avg_length = min_length = max_length = 0

            table_data.append([
                lang.upper(),
                num_samples,
                f"{avg_length:.0f}",
                min_length,
                max_length,
            ])

        headers = ["Language", "Samples", "Avg Length", "Min", "Max"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        print(f"\n{Fore.GREEN}Total Samples: {total_samples:,}")
        print(f"Total Characters: {total_chars:,}")
        print(f"Estimated Tokens (~4 chars/token): {total_chars//4:,}{Style.RESET_ALL}\n")

    def show_random_samples(self, n: int = 3):
        """Show random samples from each language."""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}RANDOM SAMPLES")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")

        for lang, texts in self.data.items():
            if not texts:
                continue

            print(f"\n{Fore.YELLOW}{lang.upper()}{Style.RESET_ALL}")
            print("-" * 70)

            samples = random.sample(texts, min(n, len(texts)))
            for i, sample in enumerate(samples, 1):
                print(f"\n{Fore.BLUE}[{i}]{Style.RESET_ALL} {sample}")

    def check_balance(self):
        """Check if dataset is balanced across languages."""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}BALANCE CHECK")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")

        counts = {lang: len(texts) for lang, texts in self.data.items()}
        total = sum(counts.values())

        if total == 0:
            print(f"{Fore.RED}No data loaded!{Style.RESET_ALL}")
            return

        table_data = []
        for lang, count in counts.items():
            percentage = (count / total) * 100
            status = "✓" if 25 <= percentage <= 40 else "⚠"
            color = Fore.GREEN if status == "✓" else Fore.YELLOW

            table_data.append([
                lang.upper(),
                count,
                f"{percentage:.1f}%",
                f"{color}{status}{Style.RESET_ALL}"
            ])

        headers = ["Language", "Count", "Percentage", "Status"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        print(f"\n{Fore.CYAN}Note: Balanced range is 25-40% per language{Style.RESET_ALL}\n")

    def check_diversity(self):
        """Check vocabulary diversity using unique word ratio."""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}DIVERSITY ANALYSIS")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")

        table_data = []
        for lang, texts in self.data.items():
            if not texts:
                continue

            # Simple word-based diversity (not perfect for all scripts)
            all_words = []
            for text in texts:
                # Split on whitespace - basic tokenization
                words = text.split()
                all_words.extend(words)

            total_words = len(all_words)
            unique_words = len(set(all_words))
            diversity_ratio = (unique_words / total_words * 100) if total_words > 0 else 0

            status = "✓" if diversity_ratio > 30 else "⚠"
            color = Fore.GREEN if status == "✓" else Fore.YELLOW

            table_data.append([
                lang.upper(),
                f"{total_words:,}",
                f"{unique_words:,}",
                f"{diversity_ratio:.1f}%",
                f"{color}{status}{Style.RESET_ALL}"
            ])

        headers = ["Language", "Total Words", "Unique Words", "Diversity", "Status"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        print(f"\n{Fore.CYAN}Note: Diversity >30% indicates good vocabulary coverage{Style.RESET_ALL}\n")

    def export_approval_report(self, output_path: str = "data/synthetic/approval_report.txt"):
        """Export a comprehensive approval report."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("SYNTHETIC DATA APPROVAL REPORT\n")
            f.write("="*70 + "\n\n")

            # Summary
            f.write("1. DATASET SUMMARY\n")
            f.write("-" * 70 + "\n")
            for lang, texts in self.data.items():
                f.write(f"{lang.upper()}: {len(texts)} samples\n")
            f.write(f"\nTotal: {sum(len(t) for t in self.data.values())} samples\n\n")

            # Balance check
            f.write("\n2. BALANCE CHECK\n")
            f.write("-" * 70 + "\n")
            total = sum(len(texts) for texts in self.data.values())
            for lang, texts in self.data.items():
                pct = (len(texts) / total * 100) if total > 0 else 0
                f.write(f"{lang.upper()}: {pct:.1f}%\n")

            # Random samples
            f.write("\n3. SAMPLE CONTENT\n")
            f.write("-" * 70 + "\n")
            for lang, texts in self.data.items():
                if not texts:
                    continue
                f.write(f"\n{lang.upper()}:\n")
                samples = random.sample(texts, min(5, len(texts)))
                for i, sample in enumerate(samples, 1):
                    f.write(f"  [{i}] {sample}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("APPROVAL STATUS: [ ] APPROVED  [ ] NEEDS REVISION\n")
            f.write("REVIEWER: _______________________\n")
            f.write("DATE: _______________________\n")
            f.write("="*70 + "\n")

        print(f"\n{Fore.GREEN}✓ Approval report exported to {output_file}{Style.RESET_ALL}\n")

    def run_full_preview(self):
        """Run complete preview workflow."""
        self.show_summary()
        self.check_balance()
        self.check_diversity()
        self.show_random_samples(n=3)
        self.export_approval_report()


if __name__ == "__main__":
    preview = DataPreview()
    preview.run_full_preview()
