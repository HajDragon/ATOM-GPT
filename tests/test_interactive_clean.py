#!/usr/bin/env python3
"""
Test the interactive chat to demonstrate that instruction leakage is fixed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the main file
from interactive_chat import generate_response, lm_enhancer

def test_interactive_responses():
    print("🧪 Testing Interactive Chat Responses (No Instruction Leakage)")
    print("=" * 60)
    
    test_prompts = [
        "write about fire",
        "tell me about death", 
        "darkness eternal",
        "metal and steel",
        "write a verse about shadows"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎤 Test {i}: '{prompt}'")
        print("🤖 ATOM-GPT: ", end="", flush=True)
        
        try:
            # Generate response exactly like the interactive chat does
            response = generate_response(prompt, max_tokens=60, temp=0.7, top_p_val=0.8, rep_penalty=1.35)
            print(response)
            
            # Check for instruction leakage
            instruction_indicators = [
                'fix grammar', 'write metal', 'dark metal style', 'words max',
                'lyrics about', 'grammar:', 'style:', 'about:', 'themed',
                '10 words', '11 words', 'max:', 'metal style'
            ]
            
            has_leakage = any(indicator in response.lower() for indicator in instruction_indicators)
            
            if has_leakage:
                print(f"   ❌ INSTRUCTION LEAKAGE DETECTED!")
                for indicator in instruction_indicators:
                    if indicator in response.lower():
                        print(f"      Found: '{indicator}'")
            else:
                print(f"   ✅ Clean response - no instruction leakage")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n📊 LM Studio Status:")
    print(f"   Available: {'✅ Yes' if lm_enhancer.available else '❌ No'}")
    if lm_enhancer.available:
        print(f"   URL: {lm_enhancer.active_url}")
    
    print("\n🎉 Test completed! All responses should be clean without instruction text.")

if __name__ == "__main__":
    test_interactive_responses()
