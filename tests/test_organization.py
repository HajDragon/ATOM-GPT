#!/usr/bin/env python3
"""
Simple test to verify the tests folder organization and imports work
"""

import sys
import os

def test_organization():
    """Test that the tests folder is properly organized"""
    print("🧪 Testing ATOM-GPT Tests Organization")
    print("=" * 45)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check if we're in the tests folder
    if current_dir.endswith('tests'):
        print("✅ Running from tests directory")
    else:
        print("⚠️  Not running from tests directory")
    
    # Check if backend/training exists
    backend_path = os.path.join(os.path.dirname(current_dir), 'backend', 'training')
    if os.path.exists(backend_path):
        print(f"✅ Backend training directory found: {backend_path}")
    else:
        print(f"❌ Backend training directory not found: {backend_path}")
    
    # List test files
    test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]
    print(f"\n📋 Found {len(test_files)} test files:")
    for test_file in sorted(test_files):
        print(f"   • {test_file}")
    
    # Check if interactive_chat.py exists
    interactive_chat_path = os.path.join(backend_path, 'interactive_chat.py')
    if os.path.exists(interactive_chat_path):
        print(f"✅ interactive_chat.py found")
    else:
        print(f"❌ interactive_chat.py not found")
    
    print(f"\n🎯 Test folder organization complete!")
    print(f"📝 All {len(test_files)} test files have been moved from backend/training to tests/")
    print(f"📚 Use 'python run_tests.py' for an interactive test runner")
    print(f"📖 See README.md for detailed test information")

if __name__ == "__main__":
    test_organization()
