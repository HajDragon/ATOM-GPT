#!/usr/bin/env python3
"""
Test script to verify conversation CRUD operations
"""

import sys
import os

# Add the current directory to the path, but don't import any AI modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
import json

BASE_URL = "http://localhost:8000"

def test_auth_and_conversations():
    """Test the authentication and conversation CRUD operations."""
    print("🧪 Testing Authentication and Conversation CRUD...")
    
    # 1. Login with admin account
    print("\n1. 🔐 Testing login...")
    login_data = {
        "email": "admin@atomgpt.local",
        "password": "admin123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    print(f"Login status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"❌ Login failed: {response.text}")
        return
    
    auth_data = response.json()
    token = auth_data["access_token"]
    print(f"✅ Login successful, got token: {token[:20]}...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Create a new conversation
    print("\n2. 📝 Testing conversation creation...")
    conv_data = {
        "title": "Test Conversation",
        "type": "chat"
    }
    
    response = requests.post(f"{BASE_URL}/api/conversations", json=conv_data, headers=headers)
    print(f"Create conversation status: {response.status_code}")
    
    if response.status_code != 201:
        print(f"❌ Conversation creation failed: {response.text}")
        return
    
    conv_response = response.json()
    conversation_id = conv_response["conversation_id"]
    print(f"✅ Conversation created: {conversation_id}")
    
    # 3. List conversations
    print("\n3. 📋 Testing conversation listing...")
    response = requests.get(f"{BASE_URL}/api/conversations", headers=headers)
    print(f"List conversations status: {response.status_code}")
    
    if response.status_code == 200:
        conversations = response.json()
        print(f"✅ Found {len(conversations.get('conversations', []))} conversations")
        if conversations.get('conversations'):
            print(f"   - Sample: {conversations['conversations'][0]['title']}")
    else:
        print(f"❌ Failed to list conversations: {response.text}")
    
    # 4. Add a message to the conversation
    print("\n4. 💬 Testing message addition...")
    message_data = {
        "role": "user",
        "content": "Hello, this is a test message!",
        "timestamp": "2025-01-09T12:00:00Z"
    }
    
    response = requests.post(f"{BASE_URL}/api/conversations/{conversation_id}/messages", 
                           json=message_data, headers=headers)
    print(f"Add message status: {response.status_code}")
    
    if response.status_code == 201:
        message_response = response.json()
        message_id = message_response["message_id"]
        print(f"✅ Message added: {message_id}")
    else:
        print(f"❌ Failed to add message: {response.text}")
    
    # 5. Get messages from the conversation
    print("\n5. 📥 Testing message retrieval...")
    response = requests.get(f"{BASE_URL}/api/conversations/{conversation_id}/messages", headers=headers)
    print(f"Get messages status: {response.status_code}")
    
    if response.status_code == 200:
        messages = response.json()
        print(f"✅ Found {len(messages.get('messages', []))} messages")
        if messages.get('messages'):
            print(f"   - Sample: {messages['messages'][0]['content'][:50]}...")
    else:
        print(f"❌ Failed to get messages: {response.text}")
    
    # 6. Update conversation title
    print("\n6. ✏️ Testing conversation update...")
    update_data = {"title": "Updated Test Conversation"}
    response = requests.patch(f"{BASE_URL}/api/conversations/{conversation_id}", 
                            json=update_data, headers=headers)
    print(f"Update conversation status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Conversation updated successfully")
    else:
        print(f"❌ Failed to update conversation: {response.text}")
    
    # 7. Delete the conversation
    print("\n7. 🗑️ Testing conversation deletion...")
    response = requests.delete(f"{BASE_URL}/api/conversations/{conversation_id}", headers=headers)
    print(f"Delete conversation status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Conversation deleted successfully")
    else:
        print(f"❌ Failed to delete conversation: {response.text}")
    
    # 8. Verify conversation is gone
    print("\n8. 🔍 Testing conversation deletion verification...")
    response = requests.get(f"{BASE_URL}/api/conversations/{conversation_id}", headers=headers)
    print(f"Get deleted conversation status: {response.status_code}")
    
    if response.status_code == 404:
        print("✅ Conversation properly deleted (404 as expected)")
    else:
        print(f"❓ Unexpected status for deleted conversation: {response.status_code}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_auth_and_conversations()
