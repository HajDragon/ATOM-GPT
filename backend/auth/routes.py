from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from backend.database.models import DatabaseManager
from datetime import timedelta
import re

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')
db = DatabaseManager()

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, ""

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['username', 'email', 'password']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        
        # Validate input
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        is_valid, password_error = validate_password(password)
        if not is_valid:
            return jsonify({'error': password_error}), 400
        
        # Check if user already exists
        if db.get_user_by_email(email):
            return jsonify({'error': 'Email already registered'}), 409
        
        # Create user
        user_id = db.create_user(
            username=username,
            email=email,
            password=password,
            first_name=data.get('first_name', ''),
            last_name=data.get('last_name', '')
        )
        
        # Create access token
        access_token = create_access_token(
            identity=user_id,
            expires_delta=timedelta(days=7)
        )
        
        # Get user data
        user = db.get_user_by_id(user_id)
        
        return jsonify({
            'access_token': access_token,
            'user': user.to_dict(),
            'message': 'Registration successful'
        }), 201
        
    except Exception as e:
        return jsonify({'error': 'Registration failed', 'details': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Authenticate user and return JWT token"""
    try:
        data = request.get_json()
        
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({'error': 'Email and password required'}), 400
        
        email = data['email'].strip().lower()
        password = data['password']
        
        # Verify user credentials
        user = db.verify_password(email, password)
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Create access token
        access_token = create_access_token(
            identity=user.id,
            expires_delta=timedelta(days=7)
        )
        
        # Log login
        db.log_api_usage(
            user_id=user.id,
            endpoint='/api/auth/login',
            method='POST',
            status_code=200,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        return jsonify({
            'access_token': access_token,
            'user': user.to_dict(),
            'message': 'Login successful'
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Login failed', 'details': str(e)}), 500

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user information"""
    try:
        user_id = get_jwt_identity()
        user = db.get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get user stats
        stats = db.get_user_stats(user_id)
        
        return jsonify({
            'user': user.to_dict(),
            'stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to get user info', 'details': str(e)}), 500

@auth_bp.route('/settings', methods=['GET'])
@jwt_required()
def get_user_settings():
    """Get user settings"""
    try:
        user_id = get_jwt_identity()
        settings = db.get_user_settings(user_id)
        return jsonify({'settings': settings}), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to get settings', 'details': str(e)}), 500

@auth_bp.route('/settings', methods=['POST'])
@jwt_required()
def update_user_settings():
    """Update user settings"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No settings provided'}), 400
        
        # Update each setting
        for key, value in data.items():
            db.set_user_setting(user_id, key, value)
        
        # Return updated settings
        settings = db.get_user_settings(user_id)
        return jsonify({'settings': settings, 'message': 'Settings updated'}), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to update settings', 'details': str(e)}), 500