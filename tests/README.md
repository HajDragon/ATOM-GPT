# ATOM-GPT Tests

This folder contains test scripts for the ATOM-GPT interactive chat system.

## Test Files

### LM Studio Enhancement Tests
- `test_clean_enhancement.py` - Tests LM Studio enhancement without instruction leakage
- `test_lm_quick.py` - Quick LM Studio enhancement validation
- `test_lm_enhancement.py` - Comprehensive LM Studio enhancement testing

### Response Generation Tests
- `test_interactive_clean.py` - Tests interactive chat responses for cleanliness
- `test_full_100_tokens.py` - Tests full response generation with 100 tokens
- `test_longer_responses.py` - Tests longer response generation capabilities
- `test_quick_tokens.py` - Tests token limit handling

### Component Tests
- `test_relevance.py` - Tests response relevance checking
- `test_completion.py` - Tests sentence completion functionality
- `test_enhancement.py` - Tests general enhancement features
- `test_simple.py` - Simple functionality tests

### Utility Tests
- `test_regex_fix.py` - Tests regex pattern fixes
- `test_sampling.py` - Tests sampling mechanisms

## Running Tests

### Quick Test Run
To run a specific test, navigate to the tests directory and run:
```bash
cd tests
python test_clean_enhancement.py
```

### Using the Test Runner
Use the test runner for an interactive experience:
```bash
cd tests
python run_tests.py
```

### Requirements
- Make sure ATOM-GPT is properly set up in the backend/training directory
- LM Studio should be running on localhost:8080 for LM Studio tests
- Tests assume the model checkpoint exists in backend/training/out-darklyrics/

## Test Categories

### ✅ Passing Tests
- LM Studio enhancement without instruction leakage
- Response generation with proper token limits
- Clean output validation

### 🔧 Development Tests
- Regex pattern testing
- Component validation
- Feature development testing

## Notes
- Tests automatically adjust import paths to work from the tests directory
- Some tests require LM Studio to be running
- Tests are designed to validate the fixes for instruction leakage and token limit issues
