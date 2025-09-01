#!/usr/bin/env python3
"""
ATOM-GPT Unified Backend
Combined authentication and AI backend server on port 8000.
"""

import os
import sys
import json
import sqlite3
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
import jwt
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from parent directory (project root)
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
except ImportError:
    print("⚠️ python-dotenv not installed. Environment variables from .env file will not be loaded.")
    print("Install with: pip install python-dotenv")

# Add necessary directories to Python path for AI imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(backend_dir, 'models'))
sys.path.append(os.path.join(backend_dir, 'training'))

# Import AI functionality
try:
    from interactive_chat import (
        generate_response, 
        generate_metal_completion,
        lm_enhancer,
        model,
        device
    )
    AI_AVAILABLE = True
    print("✅ AI modules loaded successfully")
except Exception as e:
    print(f"⚠️ AI modules not available: {e}")
    AI_AVAILABLE = False
    model = None
    device = "unavailable"
    lm_enhancer = None
    
    # Create mock functions to prevent errors
    def generate_response(message, **kwargs):
        return "AI not available. Please check backend configuration.", False
    
    def generate_metal_completion(prompt, **kwargs):
        return "AI not available. Please check backend configuration.", False

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:3001', 'http://127.0.0.1:3000', 'http://127.0.0.1:3001'], 
     methods=['GET', 'POST', 'DELETE', 'PATCH', 'OPTIONS'], 
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

# Configuration
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_EXPIRATION_HOURS'] = 24
app.config['DATABASE_PATH'] = os.path.join(os.path.dirname(__file__), 'atom_gpt.db')

# AI Status Cache
status_cache = {
    'timestamp': 0,
    'lm_studio_available': False,
    'cache_duration': 60  # Cache for 60 seconds
}

def init_database():
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            first_name VARCHAR(50),
            last_name VARCHAR(50),
            is_active BOOLEAN DEFAULT 1,
            is_admin BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id VARCHAR(100) PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title VARCHAR(255) NOT NULL,
            type VARCHAR(20) DEFAULT 'chat',
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id VARCHAR(100) PRIMARY KEY,
            conversation_id VARCHAR(100) NOT NULL,
            content TEXT NOT NULL,
            role VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tokens_used INTEGER DEFAULT 0,
            processing_time REAL DEFAULT 0,
            enhanced BOOLEAN DEFAULT 0,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    ''')
    
    # Add missing columns if they don't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE conversations ADD COLUMN is_active BOOLEAN DEFAULT 1')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute('ALTER TABLE messages ADD COLUMN tokens_used INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute('ALTER TABLE messages ADD COLUMN processing_time REAL DEFAULT 0')
    except sqlite3.OperationalError:
        pass  # Column already exists
        
    try:
        cursor.execute('ALTER TABLE messages ADD COLUMN enhanced BOOLEAN DEFAULT 0')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Create default admin user if not exists
    cursor.execute('SELECT COUNT(*) FROM users WHERE email = ?', ('admin@atomgpt.local',))
    if cursor.fetchone()[0] == 0:
        admin_password_hash = hash_password('admin123')
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, first_name, last_name, is_admin)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('admin', 'admin@atomgpt.local', admin_password_hash, 'Admin', 'User', 1))
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash a password with salt."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return salt + password_hash.hex()

def verify_password(password, password_hash):
    """Verify a password against its hash."""
    salt = password_hash[:32]
    stored_hash = password_hash[32:]
    password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return stored_hash == password_hash_check.hex()

def generate_jwt_token(user_id):
    """Generate JWT token for user."""
    payload = {
        'user_id': user_id,
        'exp': datetime.now() + timedelta(hours=app.config['JWT_EXPIRATION_HOURS'])
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_jwt_token(token):
    """Verify JWT token and return user_id."""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
        
        user_id = verify_jwt_token(token)
        if not user_id:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Get user from database
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ? AND is_active = 1', (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        # Add user to request context
        request.current_user = {
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'first_name': user[4],
            'last_name': user[5],
            'is_active': bool(user[6]),
            'is_admin': bool(user[7]),
            'created_at': user[8],
            'updated_at': user[9]
        }
        
        return f(*args, **kwargs)
    return decorated_function

@app.route('/auth/login', methods=['POST'])
def login():
    """User login endpoint."""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ? AND is_active = 1', (email,))
    user = cursor.fetchone()
    conn.close()
    
    if not user or not verify_password(password, user[3]):
        return jsonify({'error': 'Invalid email or password'}), 401
    
    token = generate_jwt_token(user[0])
    
    return jsonify({
        'access_token': token,
        'user': {
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'first_name': user[4],
            'last_name': user[5],
            'is_active': bool(user[6]),
            'is_admin': bool(user[7]),
            'created_at': user[8],
            'updated_at': user[9]
        },
        'message': 'Login successful'
    })

@app.route('/auth/register', methods=['POST'])
def register():
    """User registration endpoint."""
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    first_name = data.get('first_name', '')
    last_name = data.get('last_name', '')
    
    if not username or not email or not password:
        return jsonify({'error': 'Username, email, and password required'}), 400
    
    # Check if user already exists
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE email = ? OR username = ?', (email, username))
    if cursor.fetchone():
        conn.close()
        return jsonify({'error': 'User with this email or username already exists'}), 400
    
    # Create new user
    password_hash = hash_password(password)
    cursor.execute('''
        INSERT INTO users (username, email, password_hash, first_name, last_name)
        VALUES (?, ?, ?, ?, ?)
    ''', (username, email, password_hash, first_name, last_name))
    
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    token = generate_jwt_token(user_id)
    
    return jsonify({
        'access_token': token,
        'user': {
            'id': user_id,
            'username': username,
            'email': email,
            'first_name': first_name,
            'last_name': last_name,
            'is_active': True,
            'is_admin': False,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        },
        'message': 'Registration successful'
    }), 201

@app.route('/auth/me', methods=['GET'])
@require_auth
def get_current_user():
    """Get current user info."""
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    user_id = request.current_user['id']
    
    try:
        # Get conversation counts (only active conversations)
        cursor.execute("""
            SELECT 
                COUNT(*) as total_conversations,
                SUM(CASE WHEN type = 'chat' THEN 1 ELSE 0 END) as chat_conversations,
                SUM(CASE WHEN type = 'completion' THEN 1 ELSE 0 END) as completion_conversations
            FROM conversations 
            WHERE user_id = ? AND is_active = 1
        """, (user_id,))
        conv_stats = cursor.fetchone()
        
        # Get message counts and tokens (only from active conversations)
        cursor.execute("""
            SELECT 
                COUNT(*) as total_messages,
                SUM(COALESCE(tokens_used, 0)) as total_tokens,
                AVG(COALESCE(processing_time, 0)) as avg_response_time
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.user_id = ? AND c.is_active = 1
        """, (user_id,))
        msg_stats = cursor.fetchone()
        
        # Get enhanced requests count (only from active conversations)
        cursor.execute("""
            SELECT COUNT(*) as enhanced_requests
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.user_id = ? AND c.is_active = 1 AND m.role = 'assistant' AND m.enhanced = 1
        """, (user_id,))
        enhanced_count = cursor.fetchone()[0]
        
        # Get total API calls (user messages + assistant messages from active conversations)
        cursor.execute("""
            SELECT COUNT(*) as total_requests
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.user_id = ? AND c.is_active = 1
        """, (user_id,))
        total_requests = cursor.fetchone()[0]
        
        stats = {
            'total_conversations': conv_stats[0] or 0,
            'chat_conversations': conv_stats[1] or 0,
            'completion_conversations': conv_stats[2] or 0,
            'total_messages': msg_stats[0] or 0,
            'total_requests': total_requests or 0,
            'total_tokens': int(msg_stats[1] or 0),
            'avg_response_time': int(msg_stats[2] or 0),
            'enhanced_requests': enhanced_count or 0,
            'user_id': user_id
        }
        
    except Exception as e:
        print(f"Error calculating user stats: {e}")
        # Fallback to zeros if database query fails
        stats = {
            'total_conversations': 0,
            'chat_conversations': 0,
            'completion_conversations': 0,
            'total_messages': 0,
            'total_requests': 0,
            'total_tokens': 0,
            'avg_response_time': 0,
            'enhanced_requests': 0,
            'user_id': user_id
        }
    finally:
        conn.close()
    
    return jsonify({
        'user': request.current_user,
        'stats': stats
    })

@app.route('/conversations', methods=['GET'])
@require_auth
def get_conversations():
    """Get user conversations."""
    limit = request.args.get('limit', 50, type=int)
    conv_type = request.args.get('type')
    
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    
    # Build query based on filters
    query = '''
        SELECT c.id, c.title, c.type, c.created_at, c.updated_at,
               COUNT(m.id) as message_count,
               MAX(m.timestamp) as last_message_at
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        WHERE c.user_id = ? AND c.is_active = 1
    '''
    params = [request.current_user['id']]
    
    if conv_type:
        query += ' AND c.type = ?'
        params.append(conv_type)
    
    query += '''
        GROUP BY c.id, c.title, c.type, c.created_at, c.updated_at
        ORDER BY c.updated_at DESC
        LIMIT ?
    '''
    params.append(limit)
    
    cursor.execute(query, params)
    
    conversations = []
    for row in cursor.fetchall():
        conversations.append({
            'id': row[0],
            'user_id': request.current_user['id'],
            'title': row[1],
            'conversation_type': row[2],
            'created_at': row[3],
            'updated_at': row[4],
            'message_count': row[5] or 0,
            'last_message_at': row[6],
            'is_archived': False
        })
    
    conn.close()
    return jsonify({'conversations': conversations})

@app.route('/conversations', methods=['POST'])
@require_auth
def create_conversation():
    """Create a new conversation."""
    data = request.get_json()
    title = data.get('title', 'New Conversation')
    conv_type = data.get('type', 'chat')
    
    # Generate a more unique conversation ID
    timestamp = int(time.time() * 1000)  # milliseconds since epoch
    random_part = secrets.token_hex(8)
    conv_id = f"conv_{timestamp}_{random_part}"
    
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversations (id, user_id, title, type, is_active)
        VALUES (?, ?, ?, ?, 1)
    ''', (conv_id, request.current_user['id'], title, conv_type))
    
    conn.commit()
    conn.close()
    
    return jsonify({
        'conversation_id': conv_id,
        'id': conv_id,
        'title': title,
        'type': conv_type,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }), 201

@app.route('/conversations/<conversation_id>/messages', methods=['POST', 'GET'])
@require_auth
def handle_messages(conversation_id):
    """Handle messages - POST to add, GET to retrieve."""
    if request.method == 'POST':
        # Add a message to a conversation - verify user owns the conversation
        user_id = request.current_user['id']
        data = request.get_json()
        content = data.get('content')
        role = data.get('role')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # Optional metadata
        enhanced = data.get('enhanced', False)
        tokens_used = data.get('tokens_used', 0)
        processing_time = data.get('processing_time', 0.0)
        model_used = data.get('model_used', '')
        temperature = data.get('temperature')
        top_p = data.get('top_p')
        repetition_penalty = data.get('repetition_penalty')
        
        if not content or not role:
            return jsonify({'error': 'Content and role required'}), 400
        
        # Generate a more unique message ID
        msg_timestamp = int(time.time() * 1000)
        random_part = secrets.token_hex(6)
        message_id = f"msg_{msg_timestamp}_{random_part}"
        
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        
        # First verify the conversation belongs to the authenticated user
        cursor.execute('''
            SELECT id FROM conversations 
            WHERE id = ? AND user_id = ? AND is_active = 1
        ''', (conversation_id, user_id))
        
        if cursor.fetchone() is None:
            conn.close()
            return jsonify({'error': 'Conversation not found or access denied'}), 404
        
        cursor.execute('''
            INSERT INTO messages (id, conversation_id, content, role, timestamp, tokens_used, processing_time, enhanced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (message_id, conversation_id, content, role, timestamp, tokens_used, processing_time, enhanced))
        
        # Update conversation's updated_at timestamp
        cursor.execute('''
            UPDATE conversations 
            SET updated_at = ?
            WHERE id = ?
        ''', (timestamp, conversation_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'message_id': message_id,
            'id': message_id,
            'conversation_id': conversation_id,
            'content': content,
            'role': role,
            'timestamp': timestamp,
            'enhanced': enhanced,
            'tokens_used': tokens_used,
            'processing_time': processing_time
        }), 201
    
    elif request.method == 'GET':
        # Get messages from a conversation - verify user owns the conversation
        user_id = request.current_user['id']
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        
        # First verify the conversation belongs to the authenticated user
        cursor.execute('''
            SELECT id FROM conversations 
            WHERE id = ? AND user_id = ? AND is_active = 1
        ''', (conversation_id, user_id))
        
        if cursor.fetchone() is None:
            conn.close()
            return jsonify({'error': 'Conversation not found or access denied'}), 404
        
        # Now get the messages
        cursor.execute('''
            SELECT id, content, role, timestamp, tokens_used, processing_time, enhanced
            FROM messages 
            WHERE conversation_id = ? 
            ORDER BY timestamp ASC
        ''', (conversation_id,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'id': row[0],
                'conversation_id': conversation_id,
                'content': row[1],
                'role': row[2],
                'created_at': row[3],
                'tokens_used': row[4] or 0,
                'processing_time': row[5] or 0.0,
                'enhanced': bool(row[6]) if row[6] is not None else False
            })
        
        conn.close()
        return jsonify({'messages': messages})

@app.route('/conversations/<conversation_id>/close', methods=['POST'])
@require_auth
def close_conversation(conversation_id):
    """Close/deactivate a conversation."""
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    
    # Set conversation as inactive
    cursor.execute('''
        UPDATE conversations 
        SET is_active = 0, updated_at = ?
        WHERE id = ? AND user_id = ?
    ''', (datetime.now().isoformat(), conversation_id, request.current_user['id']))
    
    if cursor.rowcount == 0:
        conn.close()
        return jsonify({'error': 'Conversation not found'}), 404
    
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Conversation closed'})

@app.route('/conversations/<conversation_id>', methods=['DELETE'])
@require_auth
def delete_conversation(conversation_id):
    """Delete/deactivate a conversation (same as close)."""
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    
    # First verify the conversation exists and belongs to the user
    cursor.execute('''
        SELECT id FROM conversations 
        WHERE id = ? AND user_id = ? AND is_active = 1
    ''', (conversation_id, request.current_user['id']))
    
    if cursor.fetchone() is None:
        conn.close()
        return jsonify({'error': 'Conversation not found'}), 404
    
    # Set conversation as inactive
    cursor.execute('''
        UPDATE conversations 
        SET is_active = 0, updated_at = ?
        WHERE id = ? AND user_id = ?
    ''', (datetime.now().isoformat(), conversation_id, request.current_user['id']))
    
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Conversation deleted'})

@app.route('/conversations/<conversation_id>', methods=['PATCH'])
@require_auth
def update_conversation(conversation_id):
    """Update conversation details like title."""
    data = request.get_json()
    title = data.get('title')
    
    if not title:
        return jsonify({'error': 'Title is required'}), 400
    
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    
    # First verify the conversation exists and belongs to the user
    cursor.execute('''
        SELECT id FROM conversations 
        WHERE id = ? AND user_id = ? AND is_active = 1
    ''', (conversation_id, request.current_user['id']))
    
    if cursor.fetchone() is None:
        conn.close()
        return jsonify({'error': 'Conversation not found'}), 404
    
    # Update the conversation title
    cursor.execute('''
        UPDATE conversations 
        SET title = ?, updated_at = ?
        WHERE id = ? AND user_id = ?
    ''', (title, datetime.now().isoformat(), conversation_id, request.current_user['id']))
    
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Conversation updated'})

@app.route('/api/conversations', methods=['GET'])
@require_auth
def api_get_conversations():
    """API version - Get user conversations."""
    return get_conversations()

@app.route('/api/conversations', methods=['POST'])
@require_auth
def api_create_conversation():
    """API version - Create a new conversation."""
    return create_conversation()

@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
@require_auth
def api_delete_conversation(conversation_id):
    """API version - Delete/deactivate a conversation."""
    return delete_conversation(conversation_id)

@app.route('/api/conversations/<conversation_id>', methods=['PATCH'])
@require_auth
def api_update_conversation(conversation_id):
    """API version - Update conversation details."""
    return update_conversation(conversation_id)

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
@require_auth
def api_get_conversation(conversation_id):
    """API version - Get a specific conversation."""
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    
    # Verify the conversation belongs to the authenticated user
    cursor.execute('''
        SELECT c.id, c.title, c.type, c.created_at, c.updated_at,
               COUNT(m.id) as message_count,
               MAX(m.timestamp) as last_message_at
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        WHERE c.id = ? AND c.user_id = ? AND c.is_active = 1
        GROUP BY c.id, c.title, c.type, c.created_at, c.updated_at
    ''', (conversation_id, request.current_user['id']))
    
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Conversation not found'}), 404
    
    conversation = {
        'id': row[0],
        'user_id': request.current_user['id'],
        'title': row[1],
        'conversation_type': row[2],
        'created_at': row[3],
        'updated_at': row[4],
        'message_count': row[5] or 0,
        'last_message_at': row[6],
        'is_archived': False
    }
    
    conn.close()
    return jsonify({'conversation': conversation})

@app.route('/api/conversations/<conversation_id>/messages', methods=['GET', 'POST'])
@require_auth
def api_handle_messages(conversation_id):
    """API version - Handle messages."""
    return handle_messages(conversation_id)

# ============================
# AI API ROUTES (Port 8000)
# ============================

@app.route('/debug/routes', methods=['GET'])
def debug_routes():
    """Debug endpoint to list all registered routes."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': rule.rule
        })
    return jsonify({'routes': routes})

@app.route('/api/status', methods=['GET'])
@require_auth
def api_status():
    """Get AI model and LM Studio status."""
    if not AI_AVAILABLE:
        return jsonify({
            'success': False,
            'loaded': False,
            'lm_studio_available': False,
            'device': 'unavailable',
            'error': 'AI modules not loaded'
        })
    
    try:
        # Check if we're in local model only mode
        use_local_only = os.getenv('USE_LOCAL_MODEL_ONLY', 'false').lower() == 'true'
        
        if use_local_only:
            # In local model only mode, show LM Studio as disabled
            lm_studio_available = False
            lm_studio_status = "disabled (local model only)"
        else:
            current_time = time.time()
            
            # Use cached status if it's still valid
            if (current_time - status_cache['timestamp']) < status_cache['cache_duration']:
                lm_studio_available = status_cache['lm_studio_available']
            else:
                # Only check LM Studio status if cache is expired
                lm_studio_available = lm_enhancer.is_really_available() if lm_enhancer else False
                status_cache['timestamp'] = current_time
                status_cache['lm_studio_available'] = lm_studio_available
            
            lm_studio_status = "connected" if lm_studio_available else "disconnected"
        
        return jsonify({
            'success': True,
            'loaded': model is not None,
            'lm_studio_available': lm_studio_available,
            'lm_studio_status': lm_studio_status,
            'local_model_only': use_local_only,
            'device': str(device)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@require_auth
def api_chat():
    """Handle chat messages with AI and store in database."""
    if not AI_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'AI functionality not available',
            'response': 'Sorry, AI is currently unavailable.'
        }), 500
    
    try:
        data = request.get_json()
        message = data.get('message', '')
        settings = data.get('settings', {})
        conversation_id = data.get('conversation_id')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Extract settings with defaults
        max_tokens = min(settings.get('tokens', 100), 300)
        temperature = max(min(settings.get('temperature', 0.8), 2.0), 0.1)
        
        # Get user from request context (set by @require_auth)
        user_id = request.current_user['id']
        
        # Create conversation if not provided
        if not conversation_id:
            conversation_id = f"conv_{secrets.token_hex(8)}"
            title = message[:50] + "..." if len(message) > 50 else message
            
            conn = sqlite3.connect(app.config['DATABASE_PATH'])
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (id, user_id, title, type, created_at, updated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (conversation_id, user_id, title, 'chat', datetime.now().isoformat(), datetime.now().isoformat(), 1))
            conn.commit()
            conn.close()
        else:
            # Verify user owns the existing conversation
            conn = sqlite3.connect(app.config['DATABASE_PATH'])
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM conversations 
                WHERE id = ? AND user_id = ? AND is_active = 1
            ''', (conversation_id, user_id))
            
            if cursor.fetchone() is None:
                conn.close()
                return jsonify({'error': 'Conversation not found or access denied'}), 404
            conn.close()
        
        # Store user message
        user_message_id = f"msg_{secrets.token_hex(8)}"
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO messages (id, conversation_id, content, role, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_message_id, conversation_id, message, 'user', datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        print(f"Chat request: '{message}' (tokens: {max_tokens}, temp: {temperature})")
        
        start_time = time.time()
        
        # Generate response
        response, was_enhanced = generate_response(
            message, 
            max_tokens=max_tokens,
            temp=temperature
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Store AI response
        ai_message_id = f"msg_{secrets.token_hex(8)}"
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO messages (id, conversation_id, content, role, timestamp, tokens_used, processing_time, enhanced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (ai_message_id, conversation_id, response, 'assistant', datetime.now().isoformat(), 
              max_tokens, processing_time, was_enhanced))
        conn.commit()
        conn.close()
        
        print(f"Response generated in {processing_time:.2f}ms: '{response[:50]}...' (Enhanced: {was_enhanced})")
        
        return jsonify({
            'success': True,
            'response': response,
            'enhanced': was_enhanced,
            'processing_time': round(processing_time, 2),
            'conversation_id': conversation_id,
            'user_message_id': user_message_id,
            'ai_message_id': ai_message_id
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'response': 'Sorry, I encountered an error.'
        }), 500

@app.route('/api/completion', methods=['POST'])
@require_auth
def api_completion():
    """Handle text completion requests and store in database."""
    if not AI_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'AI functionality not available',
            'completion': 'Sorry, AI is currently unavailable.'
        }), 500
    
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        settings = data.get('settings', {})
        enhance_enabled = data.get('enhance', True)
        conversation_id = data.get('conversation_id')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        max_tokens = min(settings.get('tokens', 100), 300)
        temperature = max(min(settings.get('temperature', 0.8), 2.0), 0.1)
        
        # Get user from request context (set by @require_auth)
        user_id = request.current_user['id']
        
        # Create conversation if not provided
        if not conversation_id:
            conversation_id = f"conv_{secrets.token_hex(8)}"
            title = prompt[:50] + "..." if len(prompt) > 50 else prompt
            
            conn = sqlite3.connect(app.config['DATABASE_PATH'])
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (id, user_id, title, type, created_at, updated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (conversation_id, user_id, title, 'completion', datetime.now().isoformat(), datetime.now().isoformat(), 1))
            conn.commit()
            conn.close()
        else:
            # Verify user owns the existing conversation
            conn = sqlite3.connect(app.config['DATABASE_PATH'])
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM conversations 
                WHERE id = ? AND user_id = ? AND is_active = 1
            ''', (conversation_id, user_id))
            
            if cursor.fetchone() is None:
                conn.close()
                return jsonify({'error': 'Conversation not found or access denied'}), 404
            conn.close()
        
        # Store user prompt
        user_message_id = f"msg_{secrets.token_hex(8)}"
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO messages (id, conversation_id, content, role, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_message_id, conversation_id, prompt, 'user', datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        print(f"Completion request: '{prompt}' (tokens: {max_tokens}, temp: {temperature}, enhance: {enhance_enabled})")
        
        start_time = time.time()
        
        # Use specialized metal completion function
        original_completion, was_enhanced = generate_metal_completion(
            prompt,
            max_tokens=max_tokens,
            temp=temperature,
            enhance=enhance_enabled
        )
        
        # Format as prompt + completion
        completion_text = f"{prompt.strip()} {original_completion.strip()}"
        
        processing_time = (time.time() - start_time) * 1000
        
        # Store AI completion
        ai_message_id = f"msg_{secrets.token_hex(8)}"
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO messages (id, conversation_id, content, role, timestamp, tokens_used, processing_time, enhanced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (ai_message_id, conversation_id, completion_text, 'assistant', datetime.now().isoformat(),
              max_tokens, processing_time, was_enhanced))
        conn.commit()
        conn.close()
        
        print(f"Completion generated in {processing_time:.2f}ms: '{completion_text[:50]}...' (Enhanced: {was_enhanced})")
        
        return jsonify({
            'success': True,
            'completion': completion_text,
            'enhanced': was_enhanced,
            'processing_time': round(processing_time, 2),
            'conversation_id': conversation_id,
            'user_message_id': user_message_id,
            'ai_message_id': ai_message_id
        })
        
    except Exception as e:
        print(f"Completion error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'completion': 'Sorry, I encountered an error.'
        }), 500

@app.route('/api/lm-studio/status', methods=['GET'])
@require_auth
def lm_studio_status():
    """Get LM Studio connection status."""
    if not AI_AVAILABLE or not lm_enhancer:
        return jsonify({
            'success': False,
            'connected': False,
            'error': 'LM Studio functionality not available'
        })
    
    try:
        # Check if we're in local model only mode
        use_local_only = os.getenv('USE_LOCAL_MODEL_ONLY', 'false').lower() == 'true'
        
        if use_local_only:
            return jsonify({
                'success': True,
                'connected': False,
                'active_url': None,
                'status': 'disabled',
                'reason': 'Local model only mode enabled'
            })
        
        current_time = time.time()
        
        # Use cached status if it's still valid
        if (current_time - status_cache['timestamp']) < status_cache['cache_duration']:
            is_available = status_cache['lm_studio_available']
            active_url = lm_enhancer.active_url if is_available else None
        else:
            # Only check LM Studio status if cache is expired
            is_available = lm_enhancer.is_really_available()
            status_cache['timestamp'] = current_time
            status_cache['lm_studio_available'] = is_available
            active_url = lm_enhancer.active_url if is_available else None
        
        return jsonify({
            'success': True,
            'connected': is_available,
            'active_url': active_url
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/lm-studio/reconnect', methods=['POST'])
@require_auth
def lm_studio_reconnect():
    """Reconnect to LM Studio."""
    if not AI_AVAILABLE or not lm_enhancer:
        return jsonify({
            'success': False,
            'error': 'LM Studio functionality not available'
        })
    
    try:
        # Clear cache to force reconnection check
        status_cache['timestamp'] = 0
        is_available = lm_enhancer.is_really_available()
        
        return jsonify({
            'success': True,
            'connected': is_available
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'atom-gpt-backend'})

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Run the app
    port = int(os.getenv('PORT', 8000))
    debug = '--debug' in os.sys.argv
    
    print("=" * 70)
    print("🚀 ATOM-GPT UNIFIED BACKEND")
    print("=" * 70)
    print(f"🌐 Server: http://localhost:{port}")
    print(f"📊 Database: {app.config['DATABASE_PATH']}")
    print(f"🔐 Demo account: admin@atomgpt.local / admin123")
    print(f"🤖 AI Model: {'✅ Loaded' if AI_AVAILABLE and model else '❌ Not Available'}")
    print(f"🔗 LM Studio: {'✅ Available' if AI_AVAILABLE and lm_enhancer else '❌ Not Available'}")
    print(f"📱 Device: {device}")
    print("=" * 70)
    print("📋 Available Endpoints:")
    print("   Authentication: /auth/login, /auth/register, /auth/me")
    print("   Conversations: /conversations")
    print("   AI Chat: /api/chat")
    print("   AI Completion: /api/completion")
    print("   Status: /api/status, /api/lm-studio/status")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
