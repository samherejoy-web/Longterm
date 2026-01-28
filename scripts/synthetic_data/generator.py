"""Main synthetic data generation orchestrator."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List
import random

from tqdm import tqdm

from .groq_client import GroqClient
from .templates import TemplateGenerator


class SyntheticDataGenerator:
    """Orchestrate synthetic data generation with balanced distribution."""

    DOMAINS = ["general_knowledge", "cultural", "technical", "conversational"]
    LANGUAGES = ["sanskrit", "hindi", "english"]

    def __init__(self, groq_api_key: str, output_dir: str = "data/synthetic"):
        self.groq_client = GroqClient(groq_api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize template generators for each language
        self.generators = {
            lang: TemplateGenerator(lang) for lang in self.LANGUAGES
        }

    def generate_balanced_dataset(
        self,
        total_samples: int = 10000,
        refine: bool = True,
        preview_mode: bool = False,
    ) -> Dict[str, List[str]]:
        """Generate balanced dataset across languages and domains.

        Args:
            total_samples: Total number of samples to generate
            refine: Whether to refine templates with LLM
            preview_mode: If True, only generate small preview

        Returns:
            Dictionary with language -> list of texts
        """
        if preview_mode:
            total_samples = min(100, total_samples)
            print("🔍 Preview mode: Generating limited samples...")

        # Calculate samples per category (balanced distribution)
        samples_per_language = total_samples // len(self.LANGUAGES)
        samples_per_domain = samples_per_language // len(self.DOMAINS)

        print(f"\n📊 Generating {total_samples} samples:")
        print(f"  - Per language: {samples_per_language}")
        print(f"  - Per domain: {samples_per_domain}")
        print(f"  - Refinement: {'Enabled' if refine else 'Disabled'}\n")

        all_data: Dict[str, List[str]] = {lang: [] for lang in self.LANGUAGES}
        stats: Dict[str, Dict[str, int]] = {
            lang: {domain: 0 for domain in self.DOMAINS} for lang in self.LANGUAGES
        }

        # Generate for each language and domain
        for language in self.LANGUAGES:
            print(f"\n🌐 Generating {language.upper()} content...")
            generator = self.generators[language]

            for domain in self.DOMAINS:
                print(f"  📁 Domain: {domain}")

                # Generate template-based texts
                templates = generator.generate(domain, samples_per_domain)

                if refine:
                    # Refine with LLM
                    refined = []
                    for template in tqdm(templates, desc=f"    Refining {domain}"):
                        refined_text = self.groq_client.refine_text(
                            template, language, context=domain
                        )
                        if refined_text:
                            refined.append(refined_text)
                        else:
                            # Fallback to template if refinement fails
                            refined.append(template)
                    all_data[language].extend(refined)
                else:
                    all_data[language].extend(templates)

                stats[language][domain] = len(templates)

        # Add some cross-lingual mixed content (multilingual contexts)
        print("\n🔀 Generating multilingual mixed content...")
        mixed_samples = self._generate_mixed_content(samples_per_domain)
        for lang in self.LANGUAGES:
            all_data[lang].extend(mixed_samples.get(lang, []))

        # Print statistics
        self._print_statistics(stats, all_data)

        return all_data

    def _generate_mixed_content(self, count: int) -> Dict[str, List[str]]:
        """Generate content that mixes languages (common in Indian context)."""
        mixed: Dict[str, List[str]] = {lang: [] for lang in self.LANGUAGES}

        # English-Hindi code-mixing (very common)
        for _ in range(count // 3):
            templates = [
                "AI और machine learning भारत में तेजी से बढ़ रहे हैं।",
                "Programming सीखने के लिए practice और dedication चाहिए।",
                "Cloud computing ने business को transform कर दिया है।",
                "Data science में statistics और programming दोनों important हैं।",
            ]
            mixed["hindi"].append(random.choice(templates))

        # Sanskrit-English scholarly mixing
        for _ in range(count // 3):
            templates = [
                "The concept of consciousness (चेतना) is explored deeply in Vedanta.",
                "Yoga philosophy teaches mindfulness (सचेतता) and self-awareness (आत्मज्ञानम्)।",
                "Ancient Indian mathematics included the concept of zero (शून्यम्)।",
            ]
            mixed["english"].append(random.choice(templates))

        return mixed

    def _print_statistics(self, stats: Dict, all_data: Dict):
        """Print generation statistics."""
        print("\n" + "="*60)
        print("📈 GENERATION STATISTICS")
        print("="*60)

        for language in self.LANGUAGES:
            print(f"\n{language.upper()}:")
            print(f"  Total samples: {len(all_data[language])}")
            for domain, count in stats[language].items():
                percentage = (count / len(all_data[language]) * 100) if all_data[language] else 0
                print(f"  - {domain}: {count} ({percentage:.1f}%)")

        total = sum(len(texts) for texts in all_data.values())
        print(f"\n🎯 TOTAL SAMPLES: {total}")
        print("="*60 + "\n")

    def save_dataset(
        self,
        data: Dict[str, List[str]],
        format: str = "jsonl",
        preview_samples: int = 5,
    ) -> Path:
        """Save generated dataset to files.

        Args:
            data: Dictionary with language -> texts
            format: Output format (jsonl or txt)
            preview_samples: Number of samples to show in preview

        Returns:
            Path to output directory
        """
        print("\n💾 Saving dataset...")

        for language, texts in data.items():
            if format == "jsonl":
                output_file = self.output_dir / f"{language}_synthetic.jsonl"
                with open(output_file, "w", encoding="utf-8") as f:
                    for text in texts:
                        json.dump({"text": text, "language": language}, f, ensure_ascii=False)
                        f.write("\n")
            else:  # txt format
                output_file = self.output_dir / f"{language}_synthetic.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    for text in texts:
                        f.write(text + "\n\n")

            print(f"  ✅ Saved {len(texts)} samples to {output_file}")

        # Save preview
        preview_file = self.output_dir / "preview.txt"
        with open(preview_file, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("SYNTHETIC DATA PREVIEW\n")
            f.write("="*60 + "\n\n")

            for language, texts in data.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"{language.upper()} SAMPLES\n")
                f.write(f"{'='*60}\n\n")

                for i, text in enumerate(texts[:preview_samples], 1):
                    f.write(f"Sample {i}:\n{text}\n\n")

        print(f"\n📄 Preview saved to {preview_file}")
        return self.output_dir

    def generate_reasoning_chains(self, count: int = 1000) -> Dict[str, List[Dict]]:
        """Generate reasoning/thinking chains for better model reasoning."""
        print("\n🧠 Generating reasoning chains...")

        chains: Dict[str, List[Dict]] = {lang: [] for lang in self.LANGUAGES}

        prompts = {
            "english": [
                "Explain step by step how to solve: If a train travels 120 km in 2 hours, what is its speed?",
                "Think through: Why do seasons change on Earth?",
                "Reason about: What makes a good algorithm efficient?",
            ],
            "hindi": [
                "चरण दर चरण समझाएं: यदि एक ट्रेन 2 घंटे में 120 किमी चलती है, तो उसकी गति क्या है?",
                "सोचें: पृथ्वी पर ऋतुएं क्यों बदलती हैं?",
                "विचार करें: एक अच्छा algorithm efficient क्यों होता है?",
            ],
            "sanskrit": [
                "क्रमेण वद: यदि रथः द्वौ होरायाम् १२० क्रोशान् गच्छति, तर्हि तस्य वेगः कः?",
                "चिन्तयतु: पृथिव्याम् ऋतवः कस्मात् परिवर्तन्ते?",
            ]
        }

        for language, prompts_list in prompts.items():
            for prompt in tqdm(prompts_list[:count//len(prompts)], desc=f"  {language}"):
                system_prompt = f"""You are a reasoning expert. Provide step-by-step logical reasoning in {language}.
Break down the problem and show your thinking process clearly."""

                response = self.groq_client.generate(
                    prompt,
                    temperature=0.7,
                    max_tokens=1024,
                    system_prompt=system_prompt,
                )

                if response:
                    chains[language].append({
                        "question": prompt,
                        "reasoning": response,
                        "language": language,
                    })

        # Save reasoning chains
        for language, chain_list in chains.items():
            output_file = self.output_dir / f"{language}_reasoning.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for chain in chain_list:
                    json.dump(chain, f, ensure_ascii=False)
                    f.write("\n")
            print(f"  ✅ Saved {len(chain_list)} reasoning chains to {output_file}")

        return chains
