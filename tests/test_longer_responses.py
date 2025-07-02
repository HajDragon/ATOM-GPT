#!/usr/bin/env python3
"""
Test longer responses with the improved LM Studio enhancement
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interactive_chat import generate_response, lm_enhancer

def test_longer_responses():
    print("🧪 Testing Longer Response Generation")
    print("=" * 50)
    
    test_prompts = [
        "write about the eternal darkness",
        "tell me about death and shadows", 
        "create a verse about fire and steel",
        "describe the ancient metal gods"
    ]
    
    # Test with different token settings
    token_settings = [60, 80, 100]
    
    for tokens in token_settings:
        print(f"\n🎯 Testing with {tokens} max tokens:")
        print("-" * 30)
        
        for i, prompt in enumerate(test_prompts[:2], 1):  # Test first 2 prompts
            print(f"\n📝 Test {i}: '{prompt}'")
            print("🤖 ATOM-GPT: ", end="", flush=True)
            
            try:
                response = generate_response(prompt, max_tokens=tokens, temp=0.7, top_p_val=0.8, rep_penalty=1.35)
                print(response)
                
                # Show response statistics
                word_count = len(response.split())
                char_count = len(response)
                print(f"   📊 Stats: {word_count} words, {char_count} characters")
                
                # Check if LM Studio enhanced it
                if lm_enhancer.available:
                    print(f"   🔗 LM Studio: ✅ Enhanced")
                else:
                    print(f"   🔗 LM Studio: ❌ Offline")
                    
            except Exception as e:
                print(f"Error: {e}")
    
    print(f"\n📊 LM Studio Status:")
    print(f"   Available: {'✅ Yes' if lm_enhancer.available else '❌ No'}")
    if lm_enhancer.available:
        print(f"   URL: {lm_enhancer.active_url}")

if __name__ == "__main__":
    test_longer_responses()
