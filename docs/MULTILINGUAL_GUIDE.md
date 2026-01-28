# Multilingual LLM Training Guide
## Sanskrit, Hindi, English - Unbiased Synthetic Data

---

## Overview

This guide covers training your own multilingual LLM with:
- ✅ **Long-term memory** (CMS + Titans)
- ✅ **Reasoning capacity** (Self-modifying Titans)
- ✅ **Unbiased training** (Balanced synthetic data)
- ✅ **Multilingual support** (Sanskrit, Hindi, English)
- ✅ **Checkpoint management** (Save/rollback)

---

## Quick Start (4 Steps)

### 1. Generate Synthetic Data

```bash
# Generate balanced multilingual dataset
python scripts/cli/generate_data.py \
  --api-key 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 \
  --samples 10000 \
  --with-reasoning \
  --show-preview

# This will:
# - Generate 10,000 samples across Sanskrit, Hindi, English
# - Create reasoning chains for better thinking capability
# - Show preview and generate approval report
```

### 2. Preview and Validate Data

```bash
# Review data quality
python scripts/cli/preview_data.py --samples 5

# Check approval report
cat data/synthetic/approval_report.txt

# ⚠️ IMPORTANT: Review the data before proceeding!
# Make sure:
# - Balance is 25-40% per language
# - Diversity is >30%
# - Sample content looks good
```

### 3. Train Multilingual Tokenizer

```bash
# Train SentencePiece tokenizer with Devanagari support
python scripts/cli/train_multilingual_tokenizer.py \
  --vocab-size 32000 \
  --character-coverage 0.9995

# Output: artifacts/tokenizer/multilingual/multilingual_spm_32000_unigram.model
```

### 4. Start Training

```bash
# Single GPU training
python scripts/cli/train_multilingual.py train \
  --mode single \
  --device cuda:0 \
  --config multilingual_production

# OR Multi-GPU with DDP
python scripts/cli/train_multilingual.py train \
  --mode ddp \
  --num-gpus 2

# OR Multi-GPU with FSDP (for larger models)
python scripts/cli/train_multilingual.py train \
  --mode fsdp \
  --num-gpus 2
```

---

## Architecture Details

### Model Configuration (Production Scale)

```yaml
Model: HOPE (Self-Modifying Titans)
- Parameters: 1.3B (1536 dim, 32 layers)
- Vocab: 32,000 tokens (multilingual)
- Context: 2048 tokens
- Memory Hierarchy:
  * TITAN memory (32-token period)
  * CMS Fast (1-token period)
  * CMS Mid (4-token period)
  * CMS Slow (32-token period)
  * CMS Ultra (128-token period)
  * CMS Anchor (512-token period)
```

### Training Configuration

```yaml
Data:
- Total tokens: ~30B
- Languages: Sanskrit (30%), Hindi (30%), English (30%), Reasoning (10%)
- Batch size: 32
- Sequence length: 2048

Optimization:
- Optimizer: Muon (hybrid with AdamW)
- Learning rate: 1.5e-4
- Mixed precision: BF16
- Gradient checkpointing: Enabled
- torch.compile: Enabled

Schedule:
- Steps: 500,000
- Checkpoints: Every 1,000 steps
- Evaluation: As needed
```

---

## Checkpoint Management

### List Checkpoints

```bash
python scripts/cli/train_multilingual.py list \
  --run-name multilingual_production
```

### Rollback to Previous Checkpoint

```bash
# By checkpoint ID
python scripts/cli/train_multilingual.py rollback \
  --checkpoint-id multilingual_production_step_005000

# By run name and step
python scripts/cli/train_multilingual.py rollback \
  --run-name multilingual_production \
  --step 5000
```

### Create Snapshot (Backup)

```bash
python scripts/cli/train_multilingual.py snapshot \
  --run-name multilingual_production \
  --snapshot-name "before_experiment_v2"
```

### Resume Training

```bash
python scripts/cli/train_multilingual.py train \
  --resume artifacts/checkpoints/multilingual_production/step_010000.pt
```

### View Summary

```bash
python scripts/cli/train_multilingual.py summary
```

---

## Data Generation Details

### Synthetic Data Pipeline

1. **Template Generation**
   - Language-specific grammar templates
   - Domain coverage: general knowledge, cultural, technical, conversational
   - Balanced distribution across domains

2. **LLM Refinement (Groq)**
   - Llama 3.3 70B for refinement
   - Improves naturalness and coherence
   - Maintains cultural appropriateness

3. **Quality Assurance**
   - Balance checking (language distribution)
   - Diversity metrics (vocabulary coverage)
   - Manual review via approval reports

### Bias Mitigation

- **Balanced representation**: Equal coverage of all languages
- **Domain diversity**: Multiple domains prevent single-topic bias
- **Template variety**: Multiple template patterns per domain
- **LLM refinement**: Ensures natural, unbiased language
- **Manual review**: Approval reports for trainer validation

---

## Evaluation

### Zero-Shot Evaluation

```bash
# Run on all languages
python scripts/eval/zeroshot.py \
  --config configs/hope/multilingual_production.yaml \
  --checkpoint artifacts/checkpoints/multilingual_production/step_100000.pt \
  --tokenizer-path artifacts/tokenizer/multilingual/multilingual_spm_32000_unigram.model \
  --tasks all \
  --device cuda:0
```

### Long-Context Evaluation (NIAH)

```bash
python scripts/eval/niah.py \
  --config configs/hope/multilingual_production.yaml \
  --checkpoint artifacts/checkpoints/multilingual_production/step_100000.pt \
  --tokenizer-path artifacts/tokenizer/multilingual/multilingual_spm_32000_unigram.model \
  --context-lengths 2048 4096 8192
```

### Test-Time Adaptation (Memorization)

```bash
python scripts/eval/zeroshot.py \
  --config configs/hope/multilingual_production.yaml \
  --checkpoint artifacts/checkpoints/multilingual_production/step_100000.pt \
  --tokenizer-path artifacts/tokenizer/multilingual/multilingual_spm_32000_unigram.model \
  --tasks piqa,hellaswag \
  --memorize \
  --memorize-steps 2 \
  --device cuda:0
```

---

## Production Scale Training

### Hardware Requirements

**Minimum (Single GPU):**
- GPU: NVIDIA A100 40GB or H100 80GB
- RAM: 64GB
- Storage: 500GB SSD

**Recommended (Multi-GPU):**
- GPUs: 2-4x NVIDIA A100 80GB or H100 80GB
- RAM: 128GB+
- Storage: 2TB NVMe SSD
- Network: InfiniBand for multi-node

### Estimated Training Time

| Configuration | Hardware | Time (30B tokens) |
|--------------|----------|-------------------|
| Single A100 40GB | 1x A100 | ~14 days |
| Single A100 80GB | 1x A100 | ~12 days |
| DDP (2 GPUs) | 2x A100 80GB | ~6 days |
| FSDP (4 GPUs) | 4x A100 80GB | ~3 days |
| FSDP (8 GPUs) | 8x H100 80GB | ~1.5 days |

### Scaling Up

For larger datasets (100B+ tokens):

1. **Generate more data**:
   ```bash
   python scripts/cli/generate_data.py \
     --api-key <your_key> \
     --samples 100000 \
     --with-reasoning
   ```

2. **Update config**:
   - Increase `train.steps` proportionally
   - Adjust `data.mixture.samples_per_epoch`
   - Consider longer context: `data.seq_len: 4096`

3. **Use DeepSpeed ZeRO-3**:
   ```bash
   python scripts/cli/train_multilingual.py train \
     --mode deepspeed \
     --num-gpus 8
   ```

---

## Troubleshooting

### Out of Memory

- Reduce `data.batch_size`
- Enable gradient checkpointing (already enabled)
- Use FSDP with CPU offload: `train.fsdp.cpu_offload: true`
- Reduce `data.seq_len` to 1024 or 512

### Slow Training

- Ensure `train.compile.enable: true`
- Use `train.mixed_precision.enabled: true` with BF16
- Increase `data.num_workers` (default: 8)
- Use NVMe SSD for data storage

### NaN Loss

- Check `model.surprise_threshold` (set to 0.01)
- Reduce learning rate: `optim.lr: 1.0e-4`
- Check data quality (preview before training)
- Enable gradient clipping (built into Muon)

### Poor Multilingual Performance

- Check data balance (should be ~33% per language)
- Verify tokenizer coverage for all scripts
- Increase training steps
- Add more reasoning chains

---

## Best Practices

1. **Always review data** before training (use preview tool)
2. **Start with smoke test** (1000 steps) to verify setup
3. **Monitor checkpoints** regularly (every 1000 steps saved)
4. **Create snapshots** before experiments
5. **Evaluate incrementally** (every 10k steps)
6. **Use version control** for configs and scripts
7. **Document experiments** in `reports/` directory
8. **Test rollback** capability early

---

## Next Steps

1. **Generate data** with your Groq API key
2. **Review approval report** carefully
3. **Train tokenizer** with multilingual corpus
4. **Start smoke test** (1000 steps)
5. **Evaluate smoke model** on sample tasks
6. **Launch full training** after validation
7. **Monitor progress** via WandB or JSON logs
8. **Evaluate checkpoints** periodically
9. **Deploy best checkpoint** for inference

---

## Support

For issues or questions:
1. Check `docs/guide.md` for general HOPE architecture
2. Review `docs/PAPER_COMPLIANCE.md` for technical details
3. See `docs/FSDP_SCALING_GUIDE.md` for distributed training
4. Check existing issues in repository

---

## References

- Nested Learning Paper: `google_papers/Nested_Learning.pdf`
- TITANs Paper: `google_papers/TITANs.pdf`
- Original HOPE Docs: `docs/guide.md`
- Scaling Guide: `docs/scaling_guidance.md`
