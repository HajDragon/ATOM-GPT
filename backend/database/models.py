import sqlite3
import bcrypt
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class User:
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    password_hash: str = ""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Don't include password hash in serialization
        data.pop('password_hash', None)
        return data

@dataclass
class Conversation:
    id: str = ""
    user_id: int = 0
    title: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    is_archived: bool = False
    conversation_type: str = "chat"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Message:
    id: str = ""
    conversation_id: str = ""
    role: str = ""
    content: str = ""
    enhanced: bool = False
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DatabaseManager:
    def __init__(self, db_path: str = "backend/database/atom_gpt.db"):
        self.db_path = db_path
        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self):
        """Get database connection with row factory for dict-like access"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    # User Authentication Methods
    def create_user(self, username: str, email: str, password: str, **kwargs) -> int:
        """Create a new user with hashed password"""
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO users (username, email, password_hash, first_name, last_name)
                VALUES (?, ?, ?, ?, ?)
            """, (
                username, email, password_hash,
                kwargs.get('first_name'), kwargs.get('last_name')
            ))
            user_id = cursor.lastrowid
            
            # Set default settings for new user
            self._set_default_settings(user_id, conn)
            return user_id
    
    def _set_default_settings(self, user_id: int, conn):
        """Set default settings for a new user"""
        default_settings = {
            'tokens': (60, 'number'),
            'temperature': (0.7, 'number'),
            'top_p': (0.9, 'number'),
            'repetition_penalty': (1.1, 'number'),
            'theme': ('dark', 'string'),
            'auto_save': (True, 'boolean'),
            'enhance_enabled': (True, 'boolean'),
            'lm_studio_url': ('http://localhost:1234', 'string')
        }
        
        for key, (value, setting_type) in default_settings.items():
            if setting_type == 'boolean':
                value_str = str(value).lower()
            else:
                value_str = str(value)
            
            conn.execute("""
                INSERT INTO user_settings (user_id, setting_key, setting_value, setting_type)
                VALUES (?, ?, ?, ?)
            """, (user_id, key, value_str, setting_type))
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE email = ? AND is_active = TRUE", (email,))
            row = cursor.fetchone()
            if row:
                return User(**dict(row))
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE id = ? AND is_active = TRUE", (user_id,))
            row = cursor.fetchone()
            if row:
                return User(**dict(row))
        return None
    
    def verify_password(self, email: str, password: str) -> Optional[User]:
        """Verify user password and return user if valid"""
        user = self.get_user_by_email(email)
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            return user
        return None
    
    # Settings Management
    def get_user_setting(self, user_id: int, key: str, default=None):
        """Get a specific user setting"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT setting_value, setting_type FROM user_settings WHERE user_id = ? AND setting_key = ?",
                (user_id, key)
            )
            row = cursor.fetchone()
            if row:
                value, setting_type = row
                return self._parse_setting_value(value, setting_type)
        return default
    
    def set_user_setting(self, user_id: int, key: str, value: Any):
        """Set a user setting with automatic type detection"""
        setting_type, setting_value = self._serialize_setting_value(value)
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_settings 
                (user_id, setting_key, setting_value, setting_type, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (user_id, key, setting_value, setting_type))
    
    def get_user_settings(self, user_id: int) -> Dict[str, Any]:
        """Get all settings for a user"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT setting_key, setting_value, setting_type FROM user_settings WHERE user_id = ?",
                (user_id,)
            )
            settings = {}
            for row in cursor.fetchall():
                key, value, setting_type = row
                settings[key] = self._parse_setting_value(value, setting_type)
        return settings
    
    def _parse_setting_value(self, value: str, setting_type: str):
        """Parse setting value based on type"""
        if setting_type == 'json':
            return json.loads(value)
        elif setting_type == 'boolean':
            return value.lower() == 'true'
        elif setting_type == 'number':
            return float(value) if '.' in value else int(value)
        return value
    
    def _serialize_setting_value(self, value: Any):
        """Serialize setting value and determine type"""
        if isinstance(value, bool):
            return 'boolean', str(value).lower()
        elif isinstance(value, (int, float)):
            return 'number', str(value)
        elif isinstance(value, (dict, list)):
            return 'json', json.dumps(value)
        else:
            return 'string', str(value)
    
    # Conversation Management
    def create_conversation(self, user_id: int, title: str, conversation_type: str = "chat") -> str:
        """Create a new conversation"""
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO conversations (id, user_id, title, conversation_type)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, user_id, title, conversation_type))
        
        return conversation_id
    
    def get_user_conversations(self, user_id: int, limit: int = 50, conversation_type: str = None) -> List[Conversation]:
        """Get conversations for a user"""
        query = """
            SELECT * FROM conversations 
            WHERE user_id = ? AND is_archived = FALSE
        """
        params = [user_id]
        
        if conversation_type:
            query += " AND conversation_type = ?"
            params.append(conversation_type)
        
        query += " ORDER BY last_message_at DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [Conversation(**dict(row)) for row in cursor.fetchall()]
    
    def get_conversation(self, conversation_id: str, user_id: int) -> Optional[Conversation]:
        """Get a specific conversation"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id)
            )
            row = cursor.fetchone()
            if row:
                return Conversation(**dict(row))
        return None
    
    def delete_conversation(self, conversation_id: str, user_id: int) -> bool:
        """Delete a conversation and all its messages"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id)
            )
            return cursor.rowcount > 0
    
    # Message Management
    def add_message(self, conversation_id: str, role: str, content: str, **kwargs) -> str:
        """Add a message to a conversation"""
        message_id = f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        message = Message(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            enhanced=kwargs.get('enhanced', False),
            tokens_used=kwargs.get('tokens_used'),
            processing_time=kwargs.get('processing_time'),
            model_used=kwargs.get('model_used'),
            temperature=kwargs.get('temperature'),
            top_p=kwargs.get('top_p'),
            repetition_penalty=kwargs.get('repetition_penalty'),
            prompt_tokens=kwargs.get('prompt_tokens'),
            completion_tokens=kwargs.get('completion_tokens')
        )
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO messages 
                (id, conversation_id, role, content, enhanced, tokens_used, 
                 processing_time, model_used, temperature, top_p, repetition_penalty,
                 prompt_tokens, completion_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.id, message.conversation_id, message.role, message.content,
                message.enhanced, message.tokens_used, message.processing_time,
                message.model_used, message.temperature, message.top_p, 
                message.repetition_penalty, message.prompt_tokens, message.completion_tokens
            ))
            
            # Update conversation metadata
            conn.execute("""
                UPDATE conversations 
                SET last_message_at = CURRENT_TIMESTAMP,
                    message_count = message_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (conversation_id,))
        
        return message_id
    
    def get_conversation_messages(self, conversation_id: str, user_id: int) -> List[Message]:
        """Get all messages for a conversation"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT m.* FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.conversation_id = ? AND c.user_id = ?
                ORDER BY m.created_at ASC
            """, (conversation_id, user_id))
            return [Message(**dict(row)) for row in cursor.fetchall()]
    
    # Analytics and Usage Tracking
    def log_api_usage(self, user_id: int, endpoint: str, method: str, **kwargs):
        """Log API usage for analytics"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO api_usage 
                (user_id, endpoint, method, tokens_consumed, response_time, 
                 status_code, error_message, enhanced_request, user_agent, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, endpoint, method,
                kwargs.get('tokens_consumed', 0),
                kwargs.get('response_time'),
                kwargs.get('status_code', 200),
                kwargs.get('error_message'),
                kwargs.get('enhanced_request', False),
                kwargs.get('user_agent'),
                kwargs.get('ip_address')
            ))
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user statistics"""
        with self.get_connection() as conn:
            # Conversation stats
            conv_cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_conversations,
                    COUNT(CASE WHEN conversation_type = 'chat' THEN 1 END) as chat_conversations,
                    COUNT(CASE WHEN conversation_type = 'completion' THEN 1 END) as completion_conversations,
                    SUM(message_count) as total_messages
                FROM conversations 
                WHERE user_id = ? AND is_archived = FALSE
            """, (user_id,))
            conv_stats = dict(conv_cursor.fetchone())
            
            # API usage stats
            api_cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(tokens_consumed) as total_tokens,
                    AVG(response_time) as avg_response_time,
                    COUNT(CASE WHEN enhanced_request = TRUE THEN 1 END) as enhanced_requests
                FROM api_usage 
                WHERE user_id = ?
            """, (user_id,))
            api_stats = dict(api_cursor.fetchone())
            
            return {
                **conv_stats,
                **api_stats,
                'user_id': user_id
            }