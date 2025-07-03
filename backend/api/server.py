#!/usr/bin/env python3
"""
Simple Flask API Server for ATOM-GPT
"""

import sys
import os

# Add necessary directories to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(backend_dir, 'models'))
sys.path.append(os.path.join(backend_dir, 'training'))

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from datetime import datetime

# Cache for status to avoid frequent LM Studio checks
status_cache = {
    'timestamp': 0,
    'lm_studio_available': False,
    'cache_duration': 60  # Cache for 60 seconds
}

# Import from interactive_chat
from interactive_chat import (
    generate_response, 
    generate_sentence_completion,
    generate_metal_completion,  # Add the new metal completion function
    lm_enhancer,
    model,
    device
)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])  # Allow React frontend on both ports

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/api/status', methods=['GET'])
def status():
    try:
        current_time = time.time()
        
        # Use cached status if it's still valid
        if (current_time - status_cache['timestamp']) < status_cache['cache_duration']:
            lm_studio_available = status_cache['lm_studio_available']
        else:
            # Only check LM Studio status if cache is expired
            lm_studio_available = lm_enhancer.is_really_available()
            status_cache['timestamp'] = current_time
            status_cache['lm_studio_available'] = lm_studio_available
        
        return jsonify({
            'success': True,
            'loaded': model is not None,
            'lm_studio_available': lm_studio_available,
            'device': str(device)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        settings = data.get('settings', {})
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Extract settings with defaults
        max_tokens = min(settings.get('tokens', 100), 300)
        temperature = max(min(settings.get('temperature', 0.8), 2.0), 0.1)
        
        print(f"Chat request: '{message}' (tokens: {max_tokens}, temp: {temperature})")
        
        start_time = time.time()
        
        # Generate response
        response, was_enhanced = generate_response(
            message, 
            max_tokens=max_tokens,
            temp=temperature
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        print(f"Response generated in {processing_time:.2f}ms: '{response[:50]}...' (Enhanced: {was_enhanced})")
        
        return jsonify({
            'success': True,
            'response': response,
            'enhanced': was_enhanced,  # Use actual enhancement status
            'processing_time': round(processing_time, 2)
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'response': 'Sorry, I encountered an error.'
        }), 500

@app.route('/api/completion', methods=['POST'])
def completion():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        settings = data.get('settings', {})
        enhance_enabled = data.get('enhance', True)  # Default to True for backward compatibility
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        max_tokens = min(settings.get('tokens', 100), 300)
        temperature = max(min(settings.get('temperature', 0.8), 2.0), 0.1)
        
        print(f"Completion request: '{prompt}' (tokens: {max_tokens}, temp: {temperature}, enhance: {enhance_enabled})")
        
        start_time = time.time()
        
        # Use specialized metal completion function for better lyric quality
        original_completion, was_enhanced = generate_metal_completion(
            prompt,
            max_tokens=max_tokens,
            temp=temperature,
            enhance=enhance_enabled
        )
        
        # Format as prompt + completion (simple approach)
        completion_text = f"{prompt.strip()} {original_completion.strip()}"
        
        # The response from generate_response already includes enhancement if available
        
        processing_time = (time.time() - start_time) * 1000
        
        print(f"Completion generated in {processing_time:.2f}ms: '{completion_text[:50]}...' (Enhanced: {was_enhanced})")
        
        return jsonify({
            'success': True,
            'completion': completion_text,
            'enhanced': was_enhanced,  # Use actual enhancement status
            'processing_time': round(processing_time, 2)
        })
        
    except Exception as e:
        print(f"Completion error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'completion': 'Sorry, I encountered an error.'
        }), 500

@app.route('/api/lm-studio/status', methods=['GET'])
def lm_studio_status():
    try:
        current_time = time.time()
        
        # Use cached status if it's still valid
        if (current_time - status_cache['timestamp']) < status_cache['cache_duration']:
            is_available = status_cache['lm_studio_available']
        else:
            # Only check LM Studio status if cache is expired
            is_available = lm_enhancer.is_really_available()
            status_cache['timestamp'] = current_time
            status_cache['lm_studio_available'] = is_available
        
        status = lm_enhancer.get_status()
        
        return jsonify({
            'success': True,
            'connected': is_available,
            'active_url': status['active_url'] if is_available else None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("[ATOM-GPT] API Server")
    print("=" * 60)
    print(f"[Model] Loaded: {model is not None}")
    print(f"[Device] {device}")
    print(f"[LM Studio] {'Connected' if lm_enhancer.available else 'Offline'}")
    print(f"[Server] Starting on http://localhost:8001")
    print("[Frontend] React should connect to this URL")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8001, debug=True)
