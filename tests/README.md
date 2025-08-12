# ATOM-GPT Tests

Test suite validating chat generation, enhancement pipeline, sampling quality, and structural organization.

## Structure
| File | Purpose |
|------|---------|
| `run_tests.py` | Interactive runner (select tests) |
| `test_basic.py` | Baseline environment checks |
| `test_sampling.py` | Sampling parameter behavior |
| `test_completion.py` | Completion endpoint/function tests |
| `test_clean_enhancement.py` | Enhancement cleanliness (no instruction leakage) |
| `test_lm_enhancement.py` | Full enhancement evaluation |
| `test_lm_quick.py` | Fast enhancement smoke test |
| `test_interactive_clean.py` | Chat pipeline cleanliness |
| `test_full_100_tokens.py` | Longer generation stability |
| `test_longer_responses.py` | Length scaling |
| `test_quick_tokens.py` | Token limit handling |
| `test_regex_fix.py` | Cleaning regex correctness |
| `test_relevance.py` | Relevance heuristics validation |
| `test_enhancement.py` | General enhancement flag behavior |
| `test_organization.py` | Ensures test layout integrity |
| `test_simple.py` | Minimal functional sanity |

## Prerequisites
- Backend dependencies installed
- At least one trained checkpoint in `backend/out-darklyrics` (for generation tests)
- (Optional) LM Studio running on configured port for enhancement tests

## Running Tests
### Option A: Python Direct
```bash
cd tests
python test_basic.py
```

### Option B: Interactive Runner
```bash
python run_tests.py
```

### Option C: Pytest (Recommended Aggregate)
Install pytest if not present:
```bash
pip install pytest
```
Run all tests:
```bash
pytest -q tests
```
Run subset with keyword:
```bash
pytest -k enhancement -q
```

## Environment Variables for Tests
If using non-default ports:
```bash
export API_PORT=8000
export LM_STUDIO_URL=http://localhost:1234
```

## Categorization
| Category | Tests |
|----------|-------|
| Enhancement | test_clean_enhancement, test_lm_enhancement, test_lm_quick, test_enhancement |
| Generation | test_full_100_tokens, test_longer_responses, test_quick_tokens, test_interactive_clean |
| Sampling | test_sampling, test_completion |
| Quality / Cleanliness | test_regex_fix, test_relevance |
| Infra / Structure | test_basic, test_organization, test_simple |

## Adding New Tests
1. Create `test_<feature>.py`.
2. Import necessary modules (ensure `sys.path` includes project root if needed).
3. Use clear assertion messages (`assert condition, "why it failed"`).
4. If test requires checkpoint, skip gracefully if missing:
```python
import os, pytest
if not os.path.exists("../backend/out-darklyrics"):
    pytest.skip("Checkpoint dir missing")
```
5. Document purpose at top with a short docstring.

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| ImportError backend.* | Run from project root or adjust `PYTHONPATH` |
| Enhancement tests failing | Confirm LM Studio running & reachable |
| Slow tests | Limit with `pytest -k quick` |
| Random failures | Set deterministic seed in test harness |

## Related Docs
- Chat Pipeline: `../backend/training/CHAT_IMPROVEMENTS.md`
- Training Updates: `../backend/training/README_UPDATES.md`

_Last Updated: Aug 2025_
