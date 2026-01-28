#!/usr/bin/env python
"""Test script to verify multilingual training setup."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from colorama import Fore, Style, init

init(autoreset=True)


def test_imports():
    """Test that all required modules can be imported."""
    print(f"\n{Fore.CYAN}Testing imports...{Style.RESET_ALL}")
    
    try:
        import groq
        print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} groq")
    except ImportError as e:
        print(f"  {Fore.RED}\u2717{Style.RESET_ALL} groq: {e}")
        return False
    
    try:
        import sentencepiece
        print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} sentencepiece")
    except ImportError as e:
        print(f"  {Fore.RED}\u2717{Style.RESET_ALL} sentencepiece: {e}")
        return False
    
    try:
        import transformers
        print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} transformers")
    except ImportError as e:
        print(f"  {Fore.RED}\u2717{Style.RESET_ALL} transformers: {e}")
        return False
    
    try:
        from scripts.synthetic_data.groq_client import GroqClient
        print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} GroqClient")
    except ImportError as e:
        print(f"  {Fore.RED}\u2717{Style.RESET_ALL} GroqClient: {e}")
        return False
    
    try:
        from scripts.synthetic_data.templates import TemplateGenerator
        print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} TemplateGenerator")
    except ImportError as e:
        print(f"  {Fore.RED}\u2717{Style.RESET_ALL} TemplateGenerator: {e}")
        return False
    
    try:
        from scripts.checkpoint.manager import CheckpointManager
        print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} CheckpointManager")
    except ImportError as e:
        print(f"  {Fore.RED}\u2717{Style.RESET_ALL} CheckpointManager: {e}")
        return False
    
    return True


def test_template_generation():
    """Test template generation for all languages."""
    print(f"\n{Fore.CYAN}Testing template generation...{Style.RESET_ALL}")
    
    from scripts.synthetic_data.templates import TemplateGenerator
    
    languages = ["sanskrit", "hindi", "english"]
    domains = ["general_knowledge", "cultural", "technical", "conversational"]
    
    for lang in languages:
        try:
            generator = TemplateGenerator(lang)
            
            # Generate samples from each domain
            for domain in domains:
                samples = generator.generate(domain, count=2)
                if len(samples) < 2:
                    print(f"  {Fore.YELLOW}\u26a0{Style.RESET_ALL} {lang}/{domain}: Only {len(samples)} samples")
                
            print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} {lang}: All domains working")
            
        except Exception as e:
            print(f"  {Fore.RED}\u2717{Style.RESET_ALL} {lang}: {e}")
            return False
    
    return True


def test_groq_client():
    """Test Groq client initialization."""
    print(f"\n{Fore.CYAN}Testing Groq client...{Style.RESET_ALL}")
    
    from scripts.synthetic_data.groq_client import GroqClient
    
    try:
        # Test with dummy key (won't make actual API calls)
        client = GroqClient(api_key="test_key")
        print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} Client initialization")
        return True
    except Exception as e:
        print(f"  {Fore.RED}\u2717{Style.RESET_ALL} Client initialization: {e}")
        return False


def test_checkpoint_manager():
    """Test checkpoint manager."""
    print(f"\n{Fore.CYAN}Testing checkpoint manager...{Style.RESET_ALL}")
    
    from scripts.checkpoint.manager import CheckpointManager
    import tempfile
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} Manager initialization")
            
            # Test registry
            registry = manager.registry
            if "checkpoints" in registry and "active" in registry:
                print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} Registry structure")
            else:
                print(f"  {Fore.RED}\u2717{Style.RESET_ALL} Invalid registry structure")
                return False
            
        return True
    except Exception as e:
        print(f"  {Fore.RED}\u2717{Style.RESET_ALL} Checkpoint manager: {e}")
        return False


def test_configs():
    """Test that config files exist."""
    print(f"\n{Fore.CYAN}Testing configuration files...{Style.RESET_ALL}")
    
    configs = [
        "configs/hope/multilingual_production.yaml",
        "configs/data/multilingual_synthetic.yaml",
    ]
    
    all_exist = True
    for config_path in configs:
        path = Path(config_path)
        if path.exists():
            print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} {config_path}")
        else:
            print(f"  {Fore.RED}\u2717{Style.RESET_ALL} {config_path} not found")
            all_exist = False
    
    return all_exist


def test_scripts():
    """Test that CLI scripts exist and are executable."""
    print(f"\n{Fore.CYAN}Testing CLI scripts...{Style.RESET_ALL}")
    
    scripts = [
        "scripts/cli/generate_data.py",
        "scripts/cli/preview_data.py",
        "scripts/cli/train_multilingual_tokenizer.py",
        "scripts/cli/train_multilingual.py",
        "scripts/quick_start_multilingual.sh",
    ]
    
    all_exist = True
    for script_path in scripts:
        path = Path(script_path)
        if path.exists():
            # Check if executable
            if path.suffix == ".sh":
                import os
                if os.access(path, os.X_OK):
                    print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} {script_path} (executable)")
                else:
                    print(f"  {Fore.YELLOW}\u26a0{Style.RESET_ALL} {script_path} (not executable)")
            else:
                print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} {script_path}")
        else:
            print(f"  {Fore.RED}\u2717{Style.RESET_ALL} {script_path} not found")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}MULTILINGUAL LLM TRAINING - SETUP VERIFICATION")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    
    tests = [
        ("Imports", test_imports),
        ("Template Generation", test_template_generation),
        ("Groq Client", test_groq_client),
        ("Checkpoint Manager", test_checkpoint_manager),
        ("Configuration Files", test_configs),
        ("CLI Scripts", test_scripts),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n{Fore.RED}\u2717 {test_name} failed with exception: {e}{Style.RESET_ALL}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}TEST SUMMARY")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} {test_name}")
            passed += 1
        else:
            print(f"  {Fore.RED}\u2717{Style.RESET_ALL} {test_name}")
            failed += 1
    
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"Total: {len(results)} tests")
    print(f"{Fore.GREEN}Passed: {passed}{Style.RESET_ALL}")
    if failed > 0:
        print(f"{Fore.RED}Failed: {failed}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    if failed == 0:
        print(f"{Fore.GREEN}\u2705 All tests passed! Setup is complete.{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Next steps:{Style.RESET_ALL}")
        print("  1. Read MULTILINGUAL_SETUP.md for quick start")
        print("  2. Read docs/MULTILINGUAL_GUIDE.md for detailed guide")
        print("  3. Run: bash scripts/quick_start_multilingual.sh --samples 1000")
        print()
        return 0
    else:
        print(f"{Fore.RED}\u274c Some tests failed. Please check the errors above.{Style.RESET_ALL}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
