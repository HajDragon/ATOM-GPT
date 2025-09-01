# ATOM-GPT API Server

This directory contains the Flask-based API server that bridges the frontend and the trained ATOM-GPT model.

## 🚀 Quick Start

```bash
# From backend directory:
cd api
python server.py
```

The server will start on http://localhost:8000

## 📚 API Documentation

### Core Endpoints

#### POST /generate
Generate text using the ATOM-GPT model.

**Request**:
```json
{
  "prompt": "The darkness of metal",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.9
}
```

**Response**:
```json
{
  "text": "The darkness of metal envelops the soul...",
  "tokens_used": 45,
  "generation_time": 2.3
}
```

#### POST /chat
Chat interface with conversation context.

**Request**:
```json
{
  "message": "Tell me about metal music",
  "conversation_id": "optional-uuid",
  "max_tokens": 150
}
```

**Response**:
```json
{
  "response": "Metal music is a genre...",
  "conversation_id": "uuid-string",
  "message_id": "msg-uuid"
}
```

#### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "uptime": 3600
}
```

### Authentication Endpoints

#### POST /auth/register
Register a new user.

**Request**:
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

#### POST /auth/login
Authenticate user and receive JWT token.

**Request**:
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response**:
```json
{
  "token": "jwt_token_here",
  "user": {
    "id": 1,
    "email": "user@example.com"
  }
}
```

#### GET /auth/me
Get current user information (requires authentication).

**Headers**:
```
Authorization: Bearer <jwt_token>
```

### Database Endpoints

#### GET /conversations
Get user's conversations (requires authentication).

#### POST /conversations
Create new conversation (requires authentication).

#### GET /conversations/{id}/messages
Get messages for a conversation (requires authentication).

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
# Model Configuration
MODEL_PATH=../training/out-darklyrics/ckpt.pt
DEVICE=cuda  # or cpu

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Database
DATABASE_URL=sqlite:///atom_gpt.db

# Authentication
JWT_SECRET_KEY=your-secret-key-here
JWT_EXPIRATION_HOURS=24

# LM Studio Integration (Optional)
LM_STUDIO_URL=http://localhost:1234
LM_STUDIO_FALLBACK_URL=http://localhost:8080

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:3002
```

### Server Settings
Edit `settings.py` to modify:
- Default generation parameters
- Rate limiting settings
- CORS configuration
- Model loading options

## 🏗️ Architecture

### Components

- **server.py**: Main Flask application and startup
- **chat.py**: Chat interface logic and conversation management
- **completion.py**: Text completion and generation handlers
- **settings.py**: Configuration management
- **routes/**: Modular route definitions

### Request Flow

1. **Frontend** sends request to API endpoint
2. **Flask Router** directs to appropriate handler
3. **Authentication** validates JWT token (if required)
4. **Model** processes the request and generates response
5. **Database** stores conversation/message data (if applicable)
6. **Response** sent back to frontend

### Model Loading

The server automatically loads the trained model on startup:
```python
# Model loading process:
1. Check for model checkpoint in configured path
2. Load model weights and configuration
3. Move to appropriate device (GPU/CPU)
4. Set to evaluation mode
5. Ready for inference
```

## 🚨 Troubleshooting

### Common Issues

**Model not found**:
```bash
# Check model path:
ls ../training/out-darklyrics/ckpt.pt
# If missing, train a model first:
cd ../training && python train.py ../config/train_darklyrics.py
```

**CUDA out of memory**:
```python
# In settings.py, reduce batch size:
MAX_BATCH_SIZE = 1
# Or force CPU mode:
FORCE_CPU = True
```

**Port already in use**:
```bash
# Kill process on port 8000:
lsof -ti:8000 | xargs kill -9
# Or use different port:
export PORT=8001 && python server.py
```

**CORS errors**:
```python
# In settings.py, add your frontend URL:
CORS_ORIGINS = ['http://localhost:3000', 'your-frontend-url']
```

### Performance Optimization

**GPU Usage**:
```bash
# Monitor GPU usage:
nvidia-smi -l 1
# Check if model is using GPU:
curl http://localhost:8000/health
```

**Memory Management**:
```python
# In settings.py:
TORCH_CACHE_SIZE = 1024  # MB
ENABLE_MEMORY_CLEANUP = True
```

**Response Time**:
- Reduce max_tokens for faster responses
- Use lower temperature for deterministic output
- Enable model compilation for PyTorch 2.0+

## 🧪 Testing

### Manual Testing
```bash
# Health check:
curl http://localhost:8000/health

# Generate text:
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Metal is", "max_tokens": 50}'

# Test authentication:
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@atomgpt.local", "password": "admin123"}'
```

### Automated Testing
```bash
# Run API tests:
cd ../tests
python test_api.py
```

## 🔒 Security

### Authentication
- JWT tokens with configurable expiration
- Password hashing using PBKDF2
- Rate limiting on authentication endpoints

### Input Validation
- Request size limits
- Parameter validation
- SQL injection prevention

### CORS Protection
- Configurable allowed origins
- Secure headers
- Environment-based configuration

## 📊 Monitoring

### Logging
```python
# Server logs include:
- Request/response times
- Error messages and stack traces
- Model performance metrics
- Authentication events
```

### Metrics
Available at `/metrics` endpoint:
- Request count by endpoint
- Average response time
- Model inference statistics
- Error rates

## 🚀 Deployment

### Production Configuration
```python
# settings.py for production:
DEBUG = False
HOST = '0.0.0.0'
PORT = 8000
WORKERS = 4  # For gunicorn
```

### Using Gunicorn
```bash
# Install gunicorn:
pip install gunicorn

# Run with gunicorn:
gunicorn -w 4 -b 0.0.0.0:8000 server:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "server.py"]
```

## 📚 Related Documentation

- **[Backend README](../README.md)** - Complete backend setup
- **[Frontend README](../../frontend/README.md)** - Frontend integration
- **[Training Guide](../training/README_UPDATES.md)** - Model training
- **[Authentication Guide](../../AUTHENTICATION_COMPLETE.md)** - Auth system details
