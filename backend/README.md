# ATOM-GPT Backend

Backend services, model training utilities, data preparation scripts, and enhancement pipeline for ATOM-GPT (metal lyrics focused GPT fine‑tuning + optional external refinement).

## Directory Structure
```
backend/
├── api/                # Flask API (chat, completion, status, enhancement)
│   ├── server.py       # Entry point for API server
│   ├── chat.py         # Chat route handlers / logic
│   ├── completion.py   # Completion route handlers
│   ├── settings.py     # Environment + runtime settings loader
│   └── __init__.py
├── auth/               # Auth middleware & routes (if mounted)
├── config/             # Training/eval config python files + guides
│   ├── train_darklyrics.py
│   ├── test_darklyrics.py
│   └── ENHANCED_CONFIG_GUIDE.md
├── database/           # SQLite migrations / helpers (future expansion)
├── models/             # Model definition (GPT architecture)
│   └── model.py
├── scrapers/           # Dataset building & scraping utilities
│   ├── darklyrics_scraper.py
│   └── build_dataset.py
├── training/           # Training + sampling + interactive CLI
│   ├── train.py
│   ├── sample.py
│   ├── bench.py
│   ├── interactive_chat.py
│   ├── README_UPDATES.md
│   └── CHAT_IMPROVEMENTS.md
├── utils/              # Shared helpers (config, data, sorting)
├── visualization/      # Flowcharts / HTML visual aids
├── notebooks/          # Analysis & sizing notebooks
└── out-darklyrics*/    # Generated after training (checkpoints)
```

## Key Components
| Area | Files | Purpose |
|------|-------|---------|
| API Server | `api/server.py` | Serves chat & completion endpoints + status |
| Model | `models/model.py` | Lightweight GPT (DarkLyrics fine‑tune) |
| Training | `training/train.py` | Config-driven training loop |
| Sampling | `training/sample.py` | Generate lyric samples / evaluation |
| Chat Layer | `training/interactive_chat.py` | CLI + enhancement integration |
| Enhancement | `api/chat.py` + LM Studio | Optional refinement pass |
| Configuration | `config/*.py` | Reproducible experiment definitions |

## Installation (Backend Only)
From repository root:
```bash
pip install -r backend/requirements.txt
# or (if using single consolidated file)
pip install -r requirements.txt
```
If you plan to train with CUDA, install the correct PyTorch build from https://pytorch.org first.

## Environment Variables
Create a `.env` in project root (or export before launching):
```
API_HOST=0.0.0.0
API_PORT=8000
MODEL_OUT_DIR=backend/out-darklyrics
LM_STUDIO_URL=http://localhost:1234
LM_STUDIO_FALLBACK_URL=http://localhost:8080
```
`MODEL_OUT_DIR` should point to the directory containing checkpoints produced by training.

## Training Workflow
1. (Optional) Prepare dataset or use existing DarkLyrics bins.
2. Select a config in `backend/config/` (e.g. `train_darklyrics.py`).
3. Launch training:
```bash
cd backend/training
python train.py ../config/train_darklyrics.py
```
4. Checkpoints saved automatically to `../out-darklyrics` (or configured path).
5. Sample during/after training:
```bash
python sample.py --out_dir ../out-darklyrics --temperature 0.85 --top_p 0.95 --repetition_penalty 1.1
```

See `training/README_UPDATES.md` for deeper optimization tips.

## Running the API Server
After at least one checkpoint exists:
```bash
cd backend/api
python server.py
```
Default endpoints (assuming `API_PORT=8000`):
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Multi‑turn conversation (messages array) |
| POST | `/api/completion` | Single prompt completion |
| GET  | `/api/status` | Model + enhancement status, metadata |
| GET  | `/api/lm-studio/status` | External enhancement model availability |

Example chat payload:
```json
{
  "messages": [
    {"role": "user", "content": "Write a doom metal verse about ancient ruins."}
  ],
  "temperature": 0.8,
  "top_p": 0.95,
  "max_new_tokens": 140,
  "repetition_penalty": 1.1,
  "enhance": true
}
```

## Enhancement Flow (High-Level)
1. Local model produces draft.
2. Cleaning heuristics applied (see `CHAT_IMPROVEMENTS.md`).
3. If enhancement enabled & LM Studio reachable → send contextual prompt + draft.
4. Compare enhanced vs draft; accept if quality thresholds met.
5. Return final text with `enhanced` flag for frontend badge.

## Hardware Guidance
| GPU | Suggested Config |
|-----|------------------|
| 8GB | Default DarkLyrics (8L, 8H, 512 embd, 512 block) |
| 12GB | Increase batch_size or embd=640 |
| 24GB | Expand to 12L / 12H / 768 embd / block 768 |
| CPU | 4L / 4H / 384 embd / block 256 |

If you hit OOM: lower `batch_size` → raise `gradient_accumulation_steps` → reduce `block_size` → shrink `n_layer`.

## Troubleshooting
| Issue | Cause | Remedy |
|-------|-------|--------|
| Import errors | Ran scripts outside repo root | Use `cd backend/training` before running |
| 404 on API routes | Wrong working directory | Ensure you launched `api/server.py` |
| Enhancement never triggers | LM Studio not running | Start LM Studio, verify URL in `.env` |
| Repetitive loops | Low entropy settings | Increase temperature or add repetition penalty |
| Slow tokens/sec | CPU fallback | Confirm CUDA via `torch.cuda.is_available()` |

## Related Documentation
- Root Overview: `../README.md`
- Model Summary: `../ENHANCED_MODEL_SUMMARY.md`
- Config Guide: `config/ENHANCED_CONFIG_GUIDE.md`
- Chat Layer: `training/CHAT_IMPROVEMENTS.md`
- Training Updates: `training/README_UPDATES.md`

## Roadmap (Backend Scope)
- AMP / mixed precision toggle
- LoRA / PEFT adapter support
- Checkpoint manifest with git hash & config snapshot
- Automated eval metrics (perplexity, repetition ratio)
- Rhyme / meter heuristic scoring pipeline

---
_Last Updated: Aug 2025_
