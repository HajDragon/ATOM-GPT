#!/bin/bash

# ATOM-GPT Startup Script
# This script ensures proper startup of both backend and frontend

echo "======================================================================="
echo "🚀 ATOM-GPT STARTUP SCRIPT"
echo "======================================================================="

# Set LM Studio environment variable to the correct port
export LM_STUDIO_URL=http://localhost:8080
export LM_STUDIO_FALLBACK_URL=http://localhost:1234

echo "🔧 Setting LM Studio URL to: $LM_STUDIO_URL"

# Check if LM Studio is running
echo "🔍 Checking LM Studio connection..."
if curl -s http://localhost:8080/v1/models > /dev/null 2>&1; then
    echo "✅ LM Studio is running on port 8080"
else
    echo "❌ LM Studio not found on port 8080"
    echo "💡 Please start LM Studio and load a model before continuing"
    read -p "Press Enter when LM Studio is ready..."
fi

# Kill any existing backend processes
echo "🧹 Cleaning up existing processes..."
pkill -f "python app.py" 2>/dev/null || echo "No existing backend to kill"

# Start backend
echo "🚀 Starting ATOM-GPT backend..."
cd "$(dirname "$0")/backend"
python app.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check backend status
echo "🔍 Checking backend status..."
if curl -s http://localhost:8000/api/status > /dev/null 2>&1; then
    echo "✅ Backend is running on port 8000"
    
    # Test LM Studio connection through backend
    echo "🔗 Testing LM Studio integration..."
    curl -s -X POST http://localhost:8000/api/lm-studio/reconnect > /dev/null
    sleep 2
    
    LM_STATUS=$(curl -s http://localhost:8000/api/lm-studio/status | python -c "import sys, json; print(json.load(sys.stdin).get('connected', False))")
    
    if [ "$LM_STATUS" = "True" ]; then
        echo "✅ LM Studio integration working"
    else
        echo "⚠️ LM Studio integration not working - trying to reconnect..."
        curl -s -X POST http://localhost:8000/api/lm-studio/reconnect
    fi
    
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "======================================================================="
echo "🎉 ATOM-GPT is ready!"
echo "======================================================================="
echo "🌐 Backend: http://localhost:8000"
echo "🔐 Demo login: admin@atomgpt.local / admin123"
echo "🤖 AI Model: Ready"
echo "🔗 LM Studio: Connected"
echo "======================================================================="
echo "📋 API Endpoints:"
echo "   Authentication: http://localhost:8000/auth/login"
echo "   Chat: http://localhost:8000/api/chat"
echo "   Completion: http://localhost:8000/api/completion"
echo "   Status: http://localhost:8000/api/lm-studio/status"
echo "======================================================================="
echo ""
echo "💡 To test the API:"
echo "1. Get auth token: curl -X POST http://localhost:8000/auth/login -H 'Content-Type: application/json' -d '{\"email\": \"admin@atomgpt.local\", \"password\": \"admin123\"}'"
echo "2. Use the access_token in subsequent requests"
echo ""
echo "Press Ctrl+C to stop the backend"
echo ""

# Keep the script running to monitor the backend
wait $BACKEND_PID
