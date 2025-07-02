#!/usr/bin/env python3
"""
Test the improved LM Studio enhancement to verify instruction leakage is fixed
"""

import sys
import os

# Add the backend/training directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'training'))

# Import the LMStudioEnhancer class
from interactive_chat import LMStudioEnhancer

def test_enhancement_cleaning():
    print("🧪 Testing LM Studio Enhancement Cleaning...")
    
    # Initialize enhancer
    enhancer = LMStudioEnhancer()
    
    if not enhancer.available:
        print("❌ LM Studio not available - cannot test enhancement")
        return
    
    # Test cases that previously caused instruction leakage
    test_cases = [
        {
            "user_prompt": "write about fire",
            "model_response": "darkness and the void consume all",
            "expectation": "Should create fire-themed response"
        },
        {
            "user_prompt": "tell me about death", 
            "model_response": "of light Beasts will burn in an icy mountain of.",
            "expectation": "Should create death-themed response"
        },
        {
            "user_prompt": "write metal lyrics",
            "model_response": "The shadows dance eternal",
            "expectation": "Should clean up the response"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"   User: '{case['user_prompt']}'")
        print(f"   Model: '{case['model_response']}'")
        print(f"   Expected: {case['expectation']}")
        
        # Test the enhancement
        enhanced = enhancer.enhance_response(
            case['user_prompt'], 
            case['model_response'],
            max_tokens=20
        )
        
        print(f"   🔧 Enhanced: '{enhanced}'")
        
        # Check for instruction leakage
        instruction_indicators = [
            'fix grammar', 'write metal', 'dark metal style', 'words max',
            'lyrics about', 'grammar:', 'style:', 'about:', 'themed'
        ]
        
        has_leakage = any(indicator in enhanced.lower() for indicator in instruction_indicators)
        
        if has_leakage:
            print(f"   ❌ INSTRUCTION LEAKAGE DETECTED!")
        else:
            print(f"   ✅ Clean output - no instruction leakage")
        
        # Check if it's substantive
        if len(enhanced.strip()) < 5:
            print(f"   ⚠️  Response too short")
        elif enhanced == case['model_response']:
            print(f"   ℹ️  No enhancement applied (fallback)")
        else:
            print(f"   ✨ Enhancement applied successfully")

if __name__ == "__main__":
    test_enhancement_cleaning()
