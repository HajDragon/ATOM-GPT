#!/bin/bash

BASE_URL="http://localhost:8000"

echo "🧪 Testing Authentication and Conversation CRUD..."

# 1. Login
echo -e "\n1. 🔐 Testing login..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@atomgpt.local", "password": "admin123"}')

echo "Login response: $LOGIN_RESPONSE"

# Extract token from response
TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
  echo "❌ Failed to get token"
  exit 1
fi

echo "✅ Login successful, got token: ${TOKEN:0:20}..."

# 2. Create conversation
echo -e "\n2. 📝 Testing conversation creation..."
CREATE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/conversations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"title": "Test Conversation", "type": "chat"}')

echo "Create response: $CREATE_RESPONSE"

# Extract conversation ID
CONV_ID=$(echo "$CREATE_RESPONSE" | grep -o '"conversation_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$CONV_ID" ]; then
  CONV_ID=$(echo "$CREATE_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
fi

if [ -z "$CONV_ID" ]; then
  echo "❌ Failed to create conversation"
  exit 1
fi

echo "✅ Conversation created: $CONV_ID"

# 3. List conversations
echo -e "\n3. 📋 Testing conversation listing..."
LIST_RESPONSE=$(curl -s -X GET "$BASE_URL/api/conversations" \
  -H "Authorization: Bearer $TOKEN")

echo "List response: $LIST_RESPONSE"

# 4. Add message
echo -e "\n4. 💬 Testing message addition..."
MESSAGE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/conversations/$CONV_ID/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"role": "user", "content": "Hello test message!"}')

echo "Message response: $MESSAGE_RESPONSE"

# 5. Get messages
echo -e "\n5. 📥 Testing message retrieval..."
GET_MESSAGES_RESPONSE=$(curl -s -X GET "$BASE_URL/api/conversations/$CONV_ID/messages" \
  -H "Authorization: Bearer $TOKEN")

echo "Get messages response: $GET_MESSAGES_RESPONSE"

# 6. Update conversation
echo -e "\n6. ✏️ Testing conversation update..."
UPDATE_RESPONSE=$(curl -s -X PATCH "$BASE_URL/api/conversations/$CONV_ID" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"title": "Updated Test Conversation"}')

echo "Update response: $UPDATE_RESPONSE"

# 7. Delete conversation
echo -e "\n7. 🗑️ Testing conversation deletion..."
DELETE_RESPONSE=$(curl -s -X DELETE "$BASE_URL/api/conversations/$CONV_ID" \
  -H "Authorization: Bearer $TOKEN")

echo "Delete response: $DELETE_RESPONSE"

# 8. Verify deletion
echo -e "\n8. 🔍 Testing conversation deletion verification..."
VERIFY_RESPONSE=$(curl -s -X GET "$BASE_URL/api/conversations/$CONV_ID" \
  -H "Authorization: Bearer $TOKEN")

echo "Verify response: $VERIFY_RESPONSE"

echo -e "\n✅ All tests completed!"
