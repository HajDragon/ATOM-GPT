# Chat Layer Improvements & Pipeline

## Purpose
Describe the conversational layer that wraps raw model generation, adds cleaning, optional enhancement, and user/session features powering the web chat and CLI.

## Components
| Component | Location | Responsibility |
|-----------|----------|----------------|
| Conversation Buffer | Frontend (localStorage / DB) + backend session | Maintains ordered message history |
| Generation Core | `backend/training/interactive_chat.py` | Calls local model with settings |
| Cleaning Pipeline | Regex + heuristic filters | Strip artifacts, dedupe blank lines |
| Enhancement Hook | LM Studio (optional external LLM) | Refine / polish draft |
| Merge Logic | Backend enhancement module | Decide accept / fallback to draft |
| Settings Interface | Frontend UI panels | Temperature, top_p, penalties |

## Message Flow
1. User submits prompt
2. Backend builds context (recent turns within `block_size`)
3. Local model generates draft tokens (stream or buffered)
4. Cleaning pipeline normalizes whitespace, prunes low-signal noise
5. If enhancement enabled & external model reachable → send (prompt + draft)
6. Enhanced candidate returned
7. Quality heuristics compare draft vs enhanced (length ratio, reduced artifacts)
8. Selected response returned with `enhanced: true|false`
9. Conversation entry stored (future: DB persistence per user)

## Cleaning Heuristics
| Heuristic | Rationale |
|-----------|-----------|
| Collapse >2 blank lines | Avoid vertical spam |
| Trim leading punctuation drift | Remove artifacts from partial tokens |
| Remove stray unmatched quotes | Improve readability |
| Deduplicate repeated last line | Common short-loop symptom |
| Enforce max line length | Avoid runaway token repetition |

## Enhancement Acceptance Criteria (Draft → Enhanced)
| Check | Condition |
|-------|-----------|
| Length Reasonable | 0.6 ≤ enhanced_len / draft_len ≤ 1.8 |
| Artifact Reduction | Fewer obvious loops / token spam |
| No Instruction Leakage | Enhanced text lacks system prompt residue |
| Thematic Coherence | Retains metal lyrical tone |

If any critical check fails → fallback to draft.

## Sampling Controls (Frontend → Backend)
| Control | Effect |
|---------|-------|
| Temperature | Creativity vs determinism |
| Top-p | Probability nucleus truncation |
| Repetition Penalty | Penalize reusing past tokens |
| Max New Tokens | Upper bound on generation length |
| Enhancement Toggle | Enable external refinement |

## Planned Enhancements
- Multi-sample generation: produce N drafts → rank → enhance winner
- Line-level rhyme / meter scoring for targeted refinement
- Toxicity / safety filter pass pre-delivery
- Per-user adaptive temperature (historical preference)
- Lightweight embedding-based context compression (long chats)

## Debug Tips
| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| Response abruptly cut | Max tokens reached | Increase max_new_tokens |
| Excessive repetition | Low entropy settings | Raise temperature or add penalty |
| Enhancement rarely used | External model unreachable | Check LM Studio status endpoint |
| Enhancement worsens text | Heuristics too lax | Tighten acceptance thresholds |

## Testing
See `tests/test_clean_enhancement.py` and `tests/test_interactive_clean.py` for validation of cleaning and enhancement integrity.

## Related Docs
- Model Architecture: `ENHANCED_MODEL_SUMMARY.md`
- Training Updates: `backend/training/README_UPDATES.md`
- Config Guide: `backend/config/ENHANCED_CONFIG_GUIDE.md`

_Last Updated: Aug 2025_
