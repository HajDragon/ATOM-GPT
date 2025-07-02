#!/usr/bin/env python3
"""
Test script for LM Studio relevance checking
"""
import requests
import json

def test_lm_studio_relevance():
    """Test the LM Studio relevance checking functionality"""
    try:
        # Test the enhanced prompt format
        user_prompt = "write about fire"
        model_response = "darkness and void eternal"
        
        enhancement_prompt = f"""User asked: "{user_prompt}"
AI responded: "{model_response}"

Task: Check if the AI response properly addresses what the user asked for. If not, create a better metal-themed response that does. If it's relevant but poorly written, just fix the grammar and flow. Keep it dark/metal themed.

Enhanced response:"""

        payload = {
            "model": "qwen3-4b",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a metal lyrics editor. Check if AI responses match what users asked for. If not, create better metal-themed responses that do. If relevant but poorly written, fix grammar/flow. Always preserve dark/metal themes. Return ONLY the enhanced text."
                },
                {"role": "user", "content": enhancement_prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 60,
            "stream": False
        }
        
        # Try localhost first
        url = "http://localhost:8080"
        try:
            response = requests.post(
                f"{url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=15
            )
        except:
            # Try fallback URL
            url = "http://192.168.56.1:8080"
            response = requests.post(
                f"{url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=15
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ LM Studio connected at {url}")
            print(f"📝 Test Case:")
            print(f"   User: '{user_prompt}'")
            print(f"   Original: '{model_response}'")
            
            if 'choices' in result and len(result['choices']) > 0:
                enhanced = result['choices'][0]['message']['content'].strip()
                enhanced = enhanced.replace('Enhanced response:', '').strip()
                enhanced = enhanced.strip('"\'')
                print(f"   Enhanced: '{enhanced}'")
                
                if enhanced != model_response:
                    print("✅ Relevance checking working!")
                    return True
                else:
                    print("⚠️ No enhancement occurred")
                    return False
            else:
                print("❌ No response choices found")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_lm_studio_relevance()
