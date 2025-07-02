#!/usr/bin/env python3
"""
Simple test for the updated LM Studio functionality
"""
import sys
import requests
import json

def test_simple_enhancement():
    """Test LM Studio with simpler prompts"""
    try:
        # Test 1: Simple grammar fix
        payload1 = {
            "model": "qwen3-4b",
            "messages": [
                {"role": "system", "content": "You are a metal lyrics writer. Create dark, gothic, metal-themed text. Be concise and poetic."},
                {"role": "user", "content": "Fix grammar: the darkness consume all"}
            ],
            "temperature": 0.6,
            "max_tokens": 40
        }
        
        # Test 2: Create new fire response
        payload2 = {
            "model": "qwen3-4b", 
            "messages": [
                {"role": "system", "content": "You are a metal lyrics writer. Create dark, gothic, metal-themed text. Be concise and poetic."},
                {"role": "user", "content": "Write metal lyrics about: fire"}
            ],
            "temperature": 0.6,
            "max_tokens": 40
        }
        
        url = "http://localhost:8080/v1/chat/completions"
        
        print("🧪 Testing LM Studio Enhancement System")
        print("=" * 50)
        
        # Test 1
        print("\n📝 Test 1 - Grammar Fix:")
        print("   Prompt: Fix grammar: the darkness consume all")
        response1 = requests.post(url, headers={"Content-Type": "application/json"}, json=payload1, timeout=10)
        
        if response1.status_code == 200:
            result1 = response1.json()
            content1 = result1['choices'][0]['message']['content'].strip()
            print(f"   Result: {content1}")
        else:
            print(f"   Error: {response1.status_code}")
        
        # Test 2
        print("\n📝 Test 2 - Create Fire Response:")
        print("   Prompt: Write metal lyrics about: fire")
        response2 = requests.post(url, headers={"Content-Type": "application/json"}, json=payload2, timeout=10)
        
        if response2.status_code == 200:
            result2 = response2.json()
            content2 = result2['choices'][0]['message']['content'].strip()
            print(f"   Result: {content2}")
        else:
            print(f"   Error: {response2.status_code}")
        
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_simple_enhancement()
