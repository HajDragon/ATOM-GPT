#!/usr/bin/env python3
"""
Debug script to list all Flask routes
"""

import sys
import os

# Suppress AI module loading
os.environ['PYTHONPATH'] = ""

import requests

def debug_routes():
    """Test available routes."""
    print("🔍 Testing route availability...")
    
    # Test basic API endpoints
    test_urls = [
        "http://localhost:8000/auth/login",
        "http://localhost:8000/api/conversations",
        "http://localhost:8000/conversations",
        "http://localhost:8000/api/status"
    ]
    
    for url in test_urls:
        try:
            # Try both GET and POST
            get_response = requests.get(url, timeout=5)
            print(f"GET  {url}: {get_response.status_code}")
            
            post_response = requests.post(url, json={}, timeout=5)
            print(f"POST {url}: {post_response.status_code}")
            
        except Exception as e:
            print(f"ERROR {url}: {e}")
        
        print()

if __name__ == "__main__":
    debug_routes()
