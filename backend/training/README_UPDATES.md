# Training Pipeline Updates & Guide

## Purpose
This document tracks improvements to the ATOM-GPT training stack and provides guidance for configuring and running training jobs efficiently on different hardware profiles.

## Current Training Script Entry Points
| Script | Purpose |
|--------|---------|
| `train.py` | Main training loop (config driven) |
| `sample.py` | Generate text from latest/selected checkpoint |
| `bench.py` | Lightweight speed benchmark for config changes |
| `interactive_chat.py` | CLI interactive generation harness |

## Configuration Philosophy
All hyperparameters live in versioned config files under `backend/config/`. Each config is a plain Python file that sets a namespace consumed by `train.py`. This keeps experiments reproducible and diffable in git.

### Key Config Fields
| Field | Meaning | Typical DarkLyrics Value |
|-------|---------|--------------------------|
| `out_dir` | Checkpoint + log directory | `../out-darklyrics` |
| `batch_size` | Micro-batch size per gradient step | 8 (3050) |
| `gradient_accumulation_steps` | Virtual batch size multiplier | 5–8 |
| `n_layer` | Transformer layers | 8 |
| `n_head` | Attention heads | 8 |
| `n_embd` | Embedding/hidden dim | 512 |
| `block_size` | Context window (tokens) | 512 |
| `learning_rate` | Peak LR (cosine schedule) | 3e-4–6e-4 |
| `warmup_iters` | Linear warmup steps | 200 |
| `lr_decay_iters` | Steps until min LR | ~= total iters |
| `dropout` | Regularization (if overfitting) | 0.0–0.1 |

## Recent Improvements
1. Gradient Accumulation support refined for sub-8GB GPUs.
2. Centralized tokenizer reuse via GPT-2 BPE (tiktoken) for faster iteration.
3. Early sampling hook: optional periodic generation for qualitative monitoring.
4. Structured logging placeholders (easy future swap to WandB).
5. Enhanced sampling controls (temperature, top_p, repetition penalty) surfaced to frontend.
6. Modular enhancement pipeline hook (LM Studio) after local draft generation.

## Recommended Workflows
### A. Fast Sanity Test
```bash
cd backend/training
python train.py ../config/test_darklyrics.py
python sample.py --out_dir ../out-darklyrics-test
```
Expect: quick overfit, verifies environment.

### B. Full DarkLyrics Training (RTX 3050)
```bash
cd backend/training
python train.py ../config/train_darklyrics.py
```
Monitor: Loss trending from ~4.0 → 1.5–2.0.

### C. Resume Training
`train.py` auto-detects latest checkpoint in `out_dir`. Just re-run the same command.

## Sampling Tips
| Goal | Setting Adjustments |
|------|---------------------|
| Tighter structure | temperature 0.7, top_p 0.9 |
| More creativity | temperature 0.9, top_p 0.95 |
| Reduce loops | repetition_penalty 1.1 |
| Short punchy lines | max_new_tokens 60–100 |

Example:
```bash
python sample.py --out_dir ../out-darklyrics --temperature 0.85 --top_p 0.95 --repetition_penalty 1.1 --max_new_tokens 160
```

## Hardware Scaling
| GPU | Adjustments |
|-----|-------------|
| 8GB | Default config (n_layer=8, 512 embd) |
| 12GB | Raise batch_size or n_embd=640 |
| 24GB | n_layer=12, n_head=12, block_size=768 |
| CPU | Halve layers + block_size=256 |

If OOM:
1. Decrease `batch_size`.
2. Increase `gradient_accumulation_steps` to preserve effective batch.
3. Lower `block_size` before shrinking model depth.

## Mixed Precision (Planned)
AMP integration will reduce memory and accelerate training on capable GPUs. Placeholder hooks exist; implementation slated in roadmap.

## Logging (Optional WandB)
Add to config:
```python
enable_wandb = True
wandb_project = "atom-gpt"
wandb_run_name = "darklyrics_v1"
```
Then modify `train.py` (future patch) to conditionally initialize.

## Enhancement Inline vs External
Current workflow: local model produces draft → optional external LM Studio refinement → merge if quality heuristics approve. Future plan: add score-based reranking of multiple local samples before enhancement.

## Roadmap Items (Training-Specific)
- AMP toggle & GradScaler integration
- LoRA / PEFT adapter training path
- Checkpoint manifest JSON (hyperparams + git hash)
- Automated eval script (perplexity + rhyme density)
- Multi-sample reranking prior to enhancement

## Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| CUDA OOM at step 0 | Model too large | Lower `n_embd` or `batch_size` |
| Loss NaN | Exploding gradients / fp16 instability | Temporarily lower LR, disable AMP |
| Repetitive lines | Low entropy sampling | Increase temperature or add repetition penalty |
| Very slow training | CPU fallback / small batch | Verify CUDA, raise batch_size if VRAM allows |

## Related Docs
- Model Summary: `ENHANCED_MODEL_SUMMARY.md`
- Chat Layer: `CHAT_IMPROVEMENTS.md` (pipeline hooks)
- Config Guide: `backend/config/ENHANCED_CONFIG_GUIDE.md`

_Last Updated: Aug 2025_
