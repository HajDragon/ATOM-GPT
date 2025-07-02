#!/usr/bin/env python3
"""
Test full response generation with 100 tokens to verify length
"""

import sys
import os

# Add the backend/training directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'training'))

from interactive_chat import generate_response

def test_full_response():
    print("🧪 Testing Full Response Generation with 100 Tokens")
    print("=" * 55)
    
    prompts = [
        "write about the eternal darkness and shadows",
        "tell me about ancient metal warriors",
        "describe the fire that burns in the void"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Test {i}: '{prompt}'")
        print("🤖 ATOM-GPT: ", end="", flush=True)
        
        try:
            # Generate with 100 tokens like the user wants
            response = generate_response(
                prompt, 
                max_tokens=100,  # User's setting
                temp=0.7, 
                top_p_val=0.8, 
                rep_penalty=1.35
            )
            
            print(response)
            
            # Show detailed stats
            words = response.split()
            chars = len(response)
            sentences = len([s for s in response.split('.') if s.strip()])
            
            print(f"   📊 Stats: {len(words)} words | {chars} characters | {sentences} sentences")
            
            # Check if it's substantially longer than before
            if len(words) >= 8:
                print(f"   ✅ Good length - substantial response")
            elif len(words) >= 5:
                print(f"   ⚠️  Moderate length - could be longer")
            else:
                print(f"   ❌ Too short - enhancement may be over-aggressive")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n🎯 Conclusion: Responses should now be longer when you set higher token counts in the interactive chat!")

if __name__ == "__main__":
    test_full_response()
