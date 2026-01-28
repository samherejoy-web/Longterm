#!/bin/bash
# COMPLETE COMMAND REFERENCE - Copy & Paste Ready
# Multilingual LLM Training System

echo "=================================================================="
echo "  MULTILINGUAL LLM TRAINING - COMMAND REFERENCE"
echo "=================================================================="
echo ""

# ============================================
# QUICK START COMMANDS
# ============================================

echo "### QUICK START (Choose One) ###"
echo ""

echo "Option 1: Automated Pipeline (Recommended)"
echo "-------------------------------------------"
echo "bash scripts/quick_start_multilingual.sh --samples 10000 --mode single --device cuda:0"
echo ""

echo "Option 2: Step-by-Step"
echo "----------------------"
echo "# 1. Generate Data"
echo "python scripts/cli/generate_data.py \\"
echo "  --api-key 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 \\"
echo "  --samples 10000 \\"
echo "  --with-reasoning \\"
echo "  --show-preview"
echo ""
echo "# 2. Preview Data"
echo "python scripts/cli/preview_data.py --samples 5"
echo ""
echo "# 3. Train Tokenizer"
echo "python scripts/cli/train_multilingual_tokenizer.py --vocab-size 32000"
echo ""
echo "# 4. Start Training"
echo "python scripts/cli/train_multilingual.py train --mode single --device cuda:0"
echo ""

# ============================================
# DATA GENERATION
# ============================================

echo "### DATA GENERATION ###"
echo ""

echo "Smoke Test (1K samples):"
echo "python scripts/cli/generate_data.py --api-key <key> --samples 1000 --preview-only"
echo ""

echo "Medium Scale (10K samples):"
echo "python scripts/cli/generate_data.py --api-key <key> --samples 10000 --with-reasoning"
echo ""

echo "Large Scale (100K samples):"
echo "python scripts/cli/generate_data.py --api-key <key> --samples 100000 --with-reasoning"
echo ""

echo "With Custom Output Directory:"
echo "python scripts/cli/generate_data.py --api-key <key> --samples 10000 --output-dir data/synthetic_v2"
echo ""

echo "Without LLM Refinement (faster, template-only):"
echo "python scripts/cli/generate_data.py --api-key <key> --samples 10000 --no-refine"
echo ""

# ============================================
# DATA PREVIEW & VALIDATION
# ============================================

echo "### DATA PREVIEW & VALIDATION ###"
echo ""

echo "Quick Preview:"
echo "python scripts/cli/preview_data.py"
echo ""

echo "Detailed Preview (more samples):"
echo "python scripts/cli/preview_data.py --samples 10"
echo ""

echo "View Approval Report:"
echo "cat data/synthetic/approval_report.txt"
echo ""

echo "Preview Custom Directory:"
echo "python scripts/cli/preview_data.py --data-dir data/synthetic_v2"
echo ""

# ============================================
# TOKENIZER TRAINING
# ============================================

echo "### TOKENIZER TRAINING ###"
echo ""

echo "Standard (32K vocab):"
echo "python scripts/cli/train_multilingual_tokenizer.py --vocab-size 32000"
echo ""

echo "Large Vocabulary (64K):"
echo "python scripts/cli/train_multilingual_tokenizer.py --vocab-size 64000"
echo ""

echo "With Custom Output:"
echo "python scripts/cli/train_multilingual_tokenizer.py \\"
echo "  --vocab-size 32000 \\"
echo "  --output-dir artifacts/tokenizer/multilingual_v2"
echo ""

echo "BPE Model Type:"
echo "python scripts/cli/train_multilingual_tokenizer.py --model-type bpe"
echo ""

# ============================================
# TRAINING
# ============================================

echo "### TRAINING ###"
echo ""

echo "Single GPU:"
echo "python scripts/cli/train_multilingual.py train --mode single --device cuda:0"
echo ""

echo "Multi-GPU DDP (2 GPUs):"
echo "python scripts/cli/train_multilingual.py train --mode ddp --num-gpus 2"
echo ""

echo "Multi-GPU FSDP (4 GPUs):"
echo "python scripts/cli/train_multilingual.py train --mode fsdp --num-gpus 4"
echo ""

echo "DeepSpeed ZeRO-3 (8 GPUs):"
echo "python scripts/cli/train_multilingual.py train --mode deepspeed --num-gpus 8"
echo ""

echo "Smoke Test (1000 steps):"
echo "python scripts/cli/train_multilingual.py train --steps 1000 --device cuda:0"
echo ""

echo "With Custom Batch Size:"
echo "python scripts/cli/train_multilingual.py train --batch-size 16"
echo ""

echo "Resume from Checkpoint:"
echo "python scripts/cli/train_multilingual.py train \\"
echo "  --resume artifacts/checkpoints/multilingual_production/step_010000.pt"
echo ""

# ============================================
# CHECKPOINT MANAGEMENT
# ============================================

echo "### CHECKPOINT MANAGEMENT ###"
echo ""

echo "List All Checkpoints:"
echo "python scripts/cli/train_multilingual.py list"
echo ""

echo "List Specific Run:"
echo "python scripts/cli/train_multilingual.py list --run-name multilingual_production"
echo ""

echo "View Summary:"
echo "python scripts/cli/train_multilingual.py summary"
echo ""

echo "Rollback to Checkpoint ID:"
echo "python scripts/cli/train_multilingual.py rollback \\"
echo "  --checkpoint-id multilingual_production_step_005000"
echo ""

echo "Rollback to Specific Step:"
echo "python scripts/cli/train_multilingual.py rollback \\"
echo "  --run-name multilingual_production --step 5000"
echo ""

echo "Create Snapshot (Backup):"
echo "python scripts/cli/train_multilingual.py snapshot \\"
echo "  --run-name multilingual_production \\"
echo "  --snapshot-name before_experiment_v2"
echo ""

# ============================================
# EVALUATION
# ============================================

echo "### EVALUATION ###"
echo ""

echo "Zero-Shot (All Tasks):"
echo "python scripts/eval/zeroshot.py \\"
echo "  --config configs/hope/multilingual_production.yaml \\"
echo "  --checkpoint artifacts/checkpoints/multilingual_production/step_100000.pt \\"
echo "  --tokenizer-path artifacts/tokenizer/multilingual/multilingual_spm_32000_unigram.model \\"
echo "  --tasks all \\"
echo "  --device cuda:0"
echo ""

echo "Zero-Shot (Specific Tasks):"
echo "python scripts/eval/zeroshot.py \\"
echo "  --config configs/hope/multilingual_production.yaml \\"
echo "  --checkpoint artifacts/checkpoints/multilingual_production/step_100000.pt \\"
echo "  --tokenizer-path artifacts/tokenizer/multilingual/multilingual_spm_32000_unigram.model \\"
echo "  --tasks piqa,hellaswag,winogrande \\"
echo "  --device cuda:0"
echo ""

echo "With Test-Time Memorization:"
echo "python scripts/eval/zeroshot.py \\"
echo "  --config configs/hope/multilingual_production.yaml \\"
echo "  --checkpoint artifacts/checkpoints/multilingual_production/step_100000.pt \\"
echo "  --tokenizer-path artifacts/tokenizer/multilingual/multilingual_spm_32000_unigram.model \\"
echo "  --tasks piqa,hellaswag \\"
echo "  --memorize \\"
echo "  --memorize-steps 2 \\"
echo "  --device cuda:0"
echo ""

echo "Long-Context (NIAH):"
echo "python scripts/eval/niah.py \\"
echo "  --config configs/hope/multilingual_production.yaml \\"
echo "  --checkpoint artifacts/checkpoints/multilingual_production/step_100000.pt \\"
echo "  --tokenizer-path artifacts/tokenizer/multilingual/multilingual_spm_32000_unigram.model \\"
echo "  --context-lengths 2048 4096 8192"
echo ""

# ============================================
# TESTING & VERIFICATION
# ============================================

echo "### TESTING & VERIFICATION ###"
echo ""

echo "Run Setup Test:"
echo "python test_multilingual_setup.py"
echo ""

echo "Template Generation Demo:"
echo "python -c 'from scripts.synthetic_data.templates import TemplateGenerator; \\"
echo "gen = TemplateGenerator(\"hindi\"); print(gen.generate(\"general_knowledge\", 3))'"
echo ""

# ============================================
# DOCUMENTATION
# ============================================

echo "### DOCUMENTATION ###"
echo ""

echo "View Setup Guide:"
echo "cat MULTILINGUAL_SETUP.md"
echo ""

echo "View Complete Guide:"
echo "cat docs/MULTILINGUAL_GUIDE.md"
echo ""

echo "View HOPE Architecture Guide:"
echo "cat docs/guide.md"
echo ""

echo "View Distributed Training Guide:"
echo "cat docs/FSDP_SCALING_GUIDE.md"
echo ""

# ============================================
# COMMON WORKFLOWS
# ============================================

echo "### COMMON WORKFLOWS ###"
echo ""

echo "1. First Time Setup:"
echo "   python test_multilingual_setup.py"
echo "   cat MULTILINGUAL_SETUP.md"
echo ""

echo "2. Generate & Preview Data:"
echo "   python scripts/cli/generate_data.py --api-key <key> --samples 10000 --show-preview"
echo "   cat data/synthetic/approval_report.txt"
echo ""

echo "3. Train Tokenizer & Start Training:"
echo "   python scripts/cli/train_multilingual_tokenizer.py"
echo "   python scripts/cli/train_multilingual.py train --mode single"
echo ""

echo "4. Monitor Training:"
echo "   python scripts/cli/train_multilingual.py list"
echo "   python scripts/cli/train_multilingual.py summary"
echo ""

echo "5. Evaluate Checkpoint:"
echo "   python scripts/eval/zeroshot.py --config <config> --checkpoint <ckpt> --tasks all"
echo ""

echo "6. Rollback if Needed:"
echo "   python scripts/cli/train_multilingual.py rollback --run-name <run> --step <step>"
echo ""

echo "=================================================================="
echo "  For more details, see MULTILINGUAL_SETUP.md"
echo "=================================================================="
