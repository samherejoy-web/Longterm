#!/bin/bash
# Quick start script for multilingual LLM training
# This runs the complete pipeline from data generation to training

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "=========================================================================="
echo "  MULTILINGUAL LLM TRAINING - QUICK START"
echo "  Sanskrit, Hindi, English with Long-term Memory & Reasoning"
echo "=========================================================================="
echo -e "${NC}"

# Configuration
GROQ_API_KEY="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"
SAMPLES=1000  # Start with smoke test
MODE="single"  # single, ddp, fsdp, deepspeed
DEVICE="cuda:0"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --skip-tokenizer)
      SKIP_TOKENIZER=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --samples N         Number of samples to generate (default: 1000)"
      echo "  --mode MODE         Training mode: single, ddp, fsdp (default: single)"
      echo "  --device DEVICE     Device to use (default: cuda:0)"
      echo "  --skip-data         Skip data generation"
      echo "  --skip-tokenizer    Skip tokenizer training"
      echo "  -h, --help          Show this help"
      echo ""
      echo "Examples:"
      echo "  $0 --samples 10000 --mode ddp"
      echo "  $0 --skip-data --skip-tokenizer  # Jump to training"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo -e "${YELLOW}Configuration:${NC}"
echo "  Samples: ${SAMPLES}"
echo "  Mode: ${MODE}"
echo "  Device: ${DEVICE}"
echo ""

# Step 1: Generate Synthetic Data
if [ -z "$SKIP_DATA" ]; then
  echo -e "${CYAN}========================================${NC}"
  echo -e "${CYAN}Step 1: Generating Synthetic Data${NC}"
  echo -e "${CYAN}========================================${NC}"
  
  python scripts/cli/generate_data.py \
    --api-key "$GROQ_API_KEY" \
    --samples "$SAMPLES" \
    --with-reasoning \
    --reasoning-count 100 \
    --show-preview
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}Data generation failed!${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}✓ Data generation complete${NC}"
  echo ""
  
  # Show approval prompt
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}APPROVAL REQUIRED${NC}"
  echo -e "${YELLOW}========================================${NC}"
  echo ""
  echo "Please review the data approval report:"
  echo "  cat data/synthetic/approval_report.txt"
  echo ""
  echo "Or run the preview tool:"
  echo "  python scripts/cli/preview_data.py"
  echo ""
  read -p "Continue with training? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training cancelled. Review data and run again.${NC}"
    exit 0
  fi
else
  echo -e "${YELLOW}Skipping data generation...${NC}"
fi

# Step 2: Train Tokenizer
if [ -z "$SKIP_TOKENIZER" ]; then
  echo -e "${CYAN}========================================${NC}"
  echo -e "${CYAN}Step 2: Training Multilingual Tokenizer${NC}"
  echo -e "${CYAN}========================================${NC}"
  
  python scripts/cli/train_multilingual_tokenizer.py \
    --vocab-size 32000 \
    --character-coverage 0.9995
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}Tokenizer training failed!${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}✓ Tokenizer training complete${NC}"
  echo ""
else
  echo -e "${YELLOW}Skipping tokenizer training...${NC}"
fi

# Step 3: Start Training
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Step 3: Starting LLM Training${NC}"
echo -e "${CYAN}========================================${NC}"

if [ "$MODE" = "single" ]; then
  python scripts/cli/train_multilingual.py train \
    --mode single \
    --device "$DEVICE" \
    --config multilingual_production
elif [ "$MODE" = "ddp" ]; then
  python scripts/cli/train_multilingual.py train \
    --mode ddp \
    --num-gpus 2
elif [ "$MODE" = "fsdp" ]; then
  python scripts/cli/train_multilingual.py train \
    --mode fsdp \
    --num-gpus 2
else
  echo -e "${RED}Unknown mode: $MODE${NC}"
  exit 1
fi

if [ $? -eq 0 ]; then
  echo ""
  echo -e "${GREEN}==========================================================================${NC}"
  echo -e "${GREEN}  ✓ TRAINING COMPLETE!${NC}"
  echo -e "${GREEN}==========================================================================${NC}"
  echo ""
  echo "Next steps:"
  echo "  1. View checkpoints: python scripts/cli/train_multilingual.py list"
  echo "  2. Evaluate model: python scripts/eval/zeroshot.py ..."
  echo "  3. Create snapshot: python scripts/cli/train_multilingual.py snapshot ..."
  echo ""
else
  echo ""
  echo -e "${RED}Training failed or was interrupted${NC}"
  echo ""
  echo "To resume:"
  echo "  python scripts/cli/train_multilingual.py train --resume <checkpoint_path>"
  echo ""
  exit 1
fi
