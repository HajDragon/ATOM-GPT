#!/usr/bin/env python3
"""
Quick test of the LM Studio enhancement with longer tokens
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interactive_chat import lm_enhancer

def quick_test():
    print("🔧 Quick LM Studio Token Test")
    print("=" * 30)
    
    if not lm_enhancer.available:
        print("❌ LM Studio not available")
        return
    
    # Test with different max_tokens values
    test_cases = [
        {"prompt": "darkness eternal", "response": "the shadows dance", "max_tokens": 60},
        {"prompt": "write about fire", "response": "flames burn bright", "max_tokens": 100},
    ]
    
    for case in test_cases:
        print(f"\n📝 Testing with max_tokens={case['max_tokens']}")
        print(f"   Input: '{case['response']}'")
        
        enhanced = lm_enhancer.enhance_response(
            case['prompt'], 
            case['response'],
            max_tokens=case['max_tokens']
        )
        
        print(f"   Enhanced: '{enhanced}'")
        print(f"   Length: {len(enhanced)} chars, {len(enhanced.split())} words")

if __name__ == "__main__":
    quick_test()
