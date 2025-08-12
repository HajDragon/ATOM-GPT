# Enhanced Configuration Guide

## Purpose
Explain configuration surface for training, inference, enhancement, and deployment. Configs are Python files loaded by `train.py` and environment variables loaded by `python-dotenv` for runtime.

## Layers of Configuration
| Layer | Mechanism | Scope |
|-------|-----------|-------|
| Training Hyperparams | Python config files in `backend/config/` | Model + optimizer |
| Runtime Env | `.env` variables | API server + enhancement |
| Frontend Settings | Local state / API payload | Sampling per request |

## Training Config Fields
Below is a canonical example subset (see `train_darklyrics.py`).

```python
out_dir = "../out-darklyrics"
seed = 1337
batch_size = 8
gradient_accumulation_steps = 6
n_layer = 8
n_head = 8
n_embd = 512
block_size = 512
learning_rate = 6e-4
weight_decay = 1e-1
beta1, beta2 = 0.9, 0.95
warmup_iters = 200
lr_decay_iters = 60000
min_lr = 6e-5
dropout = 0.0
max_iters = 60000
compile = False  # torch.compile optional (PyTorch 2+)
```

### Adjusting for Hardware
| Constraint | Adjustment Priority |
|-----------|---------------------|
| OOM at start | Lower `batch_size` → increase `gradient_accumulation_steps` |
| Still OOM | Reduce `block_size` (costly) |
| Need faster | Enable `compile=True` (PyTorch 2) after baseline stability |
| Want larger model | Increase `n_layer` then `n_embd`, monitor VRAM |

## Optimizer & Schedule
- Optimizer: AdamW with tuned betas for transformers (`0.9, 0.95`)
- Schedule: Linear warmup → cosine decay to `min_lr`
- Gradient clipping (if enabled in future) will stabilize spikes with fp16.

## Mixed Precision (Planned)
Future flag: `use_amp = True` enabling `torch.autocast` + `GradScaler`. Save 40–50% memory on supported GPUs.

## Enhancement Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | http://localhost:1234 | Primary enhancement model endpoint |
| `LM_STUDIO_FALLBACK_URL` | http://localhost:8080 | Fallback endpoint |
| `API_HOST` | 0.0.0.0 | Backend bind address |
| `API_PORT` | 8000 | Backend port |
| `MODEL_OUT_DIR` | backend/out-darklyrics | Checkpoint directory |

Place these in `.env` or export before launching `server.py`.

## Sampling Parameters (Request-Level)
Sent per API call (chat/completion):
```json
{
  "temperature": 0.85,
  "top_p": 0.95,
  "repetition_penalty": 1.1,
  "max_new_tokens": 160,
  "enhance": true
}
```

### Recommended Ranges
| Param | Range | Notes |
|-------|-------|-------|
| temperature | 0.6–1.0 | Lower = structured, higher = creative |
| top_p | 0.85–0.97 | Nucleus size control |
| repetition_penalty | 1.05–1.25 | >1.2 may over-suppress |
| max_new_tokens | 80–220 | Longer costs more latency |

## Checkpoint Layout
```
out-darklyrics/
├── checkpoint_X.pt
├── meta.pkl
└── logs/ (optional future)
```
`train.py` loads most recent `checkpoint_*.pt` unless overridden.

## Extending Configs
1. Duplicate an existing config file.
2. Rename meaningfully: `train_darklyrics_large.py`.
3. Adjust fields; keep comments minimal and clear.
4. Commit with message referencing change rationale.

## Example: Larger Model (12GB GPU)
```python
batch_size = 8
gradient_accumulation_steps = 5
n_layer = 10
n_head = 10
n_embd = 640
block_size = 512
learning_rate = 5e-4
```

## Example: Memory Constrained (CPU / 4GB)
```python
batch_size = 4
gradient_accumulation_steps = 8
n_layer = 4
n_head = 4
n_embd = 384
block_size = 256
learning_rate = 8e-4
```

## Quality Evaluation (Upcoming)
Planned scoring utilities will compute:
- Perplexity on validation set
- Repetition ratio (unique lines / total lines)
- Rhyme density (heuristic) for metal lyric style

## Common Pitfalls
| Issue | Cause | Adjustment |
|-------|-------|-----------|
| High loss plateau >3.0 | LR too low / undertraining | Extend iters or raise LR slightly |
| Instability (NaNs) | LR too high / fp16 early | Lower LR; disable AMP until late |
| Slow tokens/sec | Small batch or CPU fallback | Confirm CUDA, adjust accumulation |

## Roadmap Flags (Coming Soon)
| Flag | Purpose |
|------|---------|
| `use_amp` | Enable mixed precision |
| `enable_wandb` | Remote experiment tracking |
| `save_interval` | Configurable checkpoint cadence |
| `eval_interval` | Automatic validation sampling |

## Related Documents
- Training Updates: `backend/training/README_UPDATES.md`
- Chat Improvements: `backend/training/CHAT_IMPROVEMENTS.md`
- Model Summary: `ENHANCED_MODEL_SUMMARY.md`

_Last Updated: Aug 2025_
