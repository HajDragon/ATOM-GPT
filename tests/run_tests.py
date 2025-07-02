#!/usr/bin/env python3
"""
Test runner for ATOM-GPT tests
This fixes import paths and runs tests from the tests directory
"""

import sys
import os

# Add the backend/training directory to the path so we can import from interactive_chat
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'training'))

def run_tests():
    """Run all available tests"""
    print("🧪 ATOM-GPT Test Suite")
    print("=" * 50)
    
    # List of available tests
    tests = [
        ("LM Studio Enhancement Cleaning", "test_clean_enhancement"),
        ("Interactive Chat Clean Responses", "test_interactive_clean"),
        ("LM Studio Quick Test", "test_lm_quick"),
        ("Full 100 Token Responses", "test_full_100_tokens"),
        ("Quick Token Limits", "test_quick_tokens"),
    ]
    
    print("Available tests:")
    for i, (name, module) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    
    print("\nSelect a test to run (1-{}) or 'all' for all tests:".format(len(tests)))
    choice = input("Choice: ").strip().lower()
    
    if choice == 'all':
        for name, module in tests:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print('='*60)
            try:
                exec(f"import {module}")
                exec(f"{module}.main()" if hasattr(exec(f"import {module}"), 'main') else f"exec(open('{module}.py').read())")
            except Exception as e:
                print(f"❌ Error running {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(tests):
        idx = int(choice) - 1
        name, module = tests[idx]
        print(f"\nRunning: {name}")
        print("-" * 40)
        try:
            exec(f"import {module}")
            exec(f"exec(open('{module}.py').read())")
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    # Change to the tests directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_tests()
