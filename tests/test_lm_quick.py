#!/usr/bin/env python3
"""
Simple test to check a quick LM Studio interaction without full interactive mode
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interactive_chat import lm_enhancer

def quick_lm_test():
    print("🔧 Quick LM Studio Enhancement Test")
    print("=" * 40)
    
    if not lm_enhancer.available:
        print("❌ LM Studio not available")
        return
    
    # Test the /lmstudio test functionality
    print("\n📝 Testing relevance and enhancement:")
    
    # Test 1: Grammar fix (relevant)
    result1 = lm_enhancer.enhance_response("darkness", "the darkness consume all", 30)
    print(f"   Input: 'the darkness consume all'")
    print(f"   Output: '{result1}'")
    print(f"   Relevant: {lm_enhancer._is_response_relevant('darkness', 'the darkness consume all')}")
    
    # Test 2: New response (irrelevant)
    result2 = lm_enhancer.enhance_response("write about fire", "darkness and void eternal", 30)
    print(f"\n   Input: 'darkness and void eternal'")
    print(f"   Output: '{result2}'")
    print(f"   Relevant: {lm_enhancer._is_response_relevant('write about fire', 'darkness and void eternal')}")
    
    # Check for instruction leakage
    responses = [result1, result2]
    for i, response in enumerate(responses, 1):
        instruction_indicators = [
            'fix grammar', 'write metal', 'dark metal style', 'words max',
            'lyrics about', 'grammar:', 'style:', 'about:', 'themed',
            '10 words', '11 words', 'max:', 'metal style'
        ]
        
        has_leakage = any(indicator in response.lower() for indicator in instruction_indicators)
        
        if has_leakage:
            print(f"   ❌ Response {i} has instruction leakage!")
        else:
            print(f"   ✅ Response {i} is clean")
    
    print("\n🎉 LM Studio test complete!")

if __name__ == "__main__":
    quick_lm_test()
