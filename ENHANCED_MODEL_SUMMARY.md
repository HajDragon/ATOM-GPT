# Enhanced Model Summary

## Overview
ATOM-GPT focuses on generating high-quality metal lyrics using a lightweight GPT architecture fine‑tuned on the DarkLyrics corpus. It supports two operation modes:
- Base Model: Local GPT (trained with backend/training/train.py)
- Enhanced Mode: Local output post-processed / optionally augmented via LM Studio (external LLM) for refinement

## Current Model (DarkLyrics Config)
| Component | Value |
|-----------|-------|
| Layers (n_layer) | 8 |
| Heads (n_head) | 8 |
| Embedding dim (n_embd) | 512 |
| Context length (block_size) | 512 tokens |
| Vocab | Byte-level BPE (tiktoken / GPT-2) |
| Params (approx) | ~45M |
| Precision | fp32 (recommended: fp16 w/ AMP if GPU supports) |
| Optimizer | AdamW (betas tuned for transformer training) |
| LR schedule | Warmup + cosine decay |
| Dataset | Curated metal lyrics (DarkLyrics) |

## Training Enhancements
1. Config Driven: All hyperparameters consolidated in backend/config/*.py for reproducibility.
2. Tokenization: Reuse GPT-2 tokenizer for compatibility with external tooling and faster experimentation.
3. Gradient Accumulation: Simulates larger batch sizes on 8GB GPUs.
4. Checkpointing: Periodic saving to backend/training/out-darklyrics for resume + sampling.
5. Sampling Utilities: sample.py supports temperature, top_p, repetition penalties (frontend exposes controls).

## Inference & Chat Layer
The chat system wraps raw model generations with:
- Conversation Buffer: Maintains user/assistant turns (frontend + backend synergy)
- Cleaning Pipeline: Regex & heuristic filters to reduce artifacts and leakage
- Enhancement Hook: Optional LM Studio call -> merges enhanced segments if quality improves
- Safety / Length Control: Max token limits + truncation safeguards

## Enhancement Flow (When Enabled)
1. User prompt -> Local model draft
2. Draft quality heuristics (length, coherence markers)
3. If enabled & LM Studio reachable: send combined context for refinement
4. Merge enhanced response; mark with enhancement flag for UI badge
5. Store conversation metadata (future: DB persistence per user)

## Token & Memory Optimization
| Technique | Purpose |
|-----------|---------|
| Gradient Accumulation | Fit effective larger batches |
| Smaller Embedding Dim | Keep VRAM <8GB |
| Adjustable block_size | Reduce memory when OOM |
| Early Sampling | Quick quality validation before long runs |

## Suggested Hardware Profiles
| GPU | Adjustments |
|-----|-------------|
| 8GB (RTX 3050) | Default config, keep n_layer=8 |
| 12GB (3060) | Increase batch_size or n_embd=640 |
| 24GB (4090) | n_layer=12, n_head=12, n_embd=768, block_size=768 |
| CPU Only | Reduce to n_layer=4, n_embd=384, block_size=256 |

## Quality Tips
- Lower temperature (0.7) for structured verse forms
- Higher top_p (0.95) for creative, surreal imagery
- Apply repetition penalty if loops appear (1.1–1.2)
- Fine‑tune on subgenre subsets for stylistic control (e.g., doom vs. black metal)

## Extending the Model
| Goal | Action |
|------|--------|
| New Dataset | Duplicate prepare script; generate train/val bins |
| Larger Context | Increase block_size then retrain from scratch |
| Style Control | Add special tokens per subgenre; fine‑tune |
| Faster Inference | Export to TorchScript or ONNX; enable fp16 |
| Instructional Use | Add instruction-format dataset; fine‑tune with chat turns |

## Roadmap
- Mixed Precision (AMP) toggle in training scripts
- LoRA / PEFT adapters for rapid style adaptation
- Checkpoint version metadata auto-logging (JSON manifest)
- Per-user conversation persistence + retrieval augmentation
- Advanced rhyme / meter scoring for guided generation

## Related Docs
- Training Updates: backend/training/README_UPDATES.md
- Chat Improvements: backend/training/CHAT_IMPROVEMENTS.md
- Config Guide: backend/config/ENHANCED_CONFIG_GUIDE.md

Last Updated: Aug 2025
