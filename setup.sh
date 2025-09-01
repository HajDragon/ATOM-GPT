#!/bin/bash
# ============================
# ATOM-GPT Quick Setup Script
# ============================
# This script sets up the ATOM-GPT project for development

set -e  # Exit on any error

echo "🚀 ATOM-GPT Quick Setup"
echo "======================="

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the ATOM-GPT project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
echo "🔍 Checking required tools..."

if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed"
    echo "   Please install Python 3.8+ from https://python.org"
    exit 1
fi

if ! command_exists node; then
    echo "❌ Node.js is required but not installed"
    echo "   Please install Node.js 16+ from https://nodejs.org"
    exit 1
fi

if ! command_exists npm; then
    echo "❌ npm is required but not installed"
    echo "   npm usually comes with Node.js"
    exit 1
fi

echo "✅ All required tools found"

# Setup Python virtual environment
echo ""
echo "🐍 Setting up Python environment..."

if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

echo "   Activating virtual environment..."
source venv/bin/activate

echo "   Upgrading pip..."
pip install --upgrade pip

echo "   Installing Python dependencies..."
pip install -r requirements.txt

# Setup backend additional dependencies
echo "   Installing backend dependencies..."
cd backend
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
cd ..

# Setup frontend
echo ""
echo "⚛️ Setting up Frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "   Installing npm dependencies..."
    npm install
else
    echo "   Updating npm dependencies..."
    npm update
fi

echo "   Building frontend..."
npm run build

cd ..

# Create .env file if it doesn't exist
echo ""
echo "🔧 Setting up configuration..."

if [ ! -f ".env" ]; then
    echo "   Creating .env file..."
    cat > .env << EOL
# ATOM-GPT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
PORT=8000
NODE_ENV=development

# AI Configuration
USE_LOCAL_MODEL_ONLY=false
MAX_TOKENS=300
DEFAULT_TEMPERATURE=0.8

# LM Studio Configuration (optional)
LM_STUDIO_URL=http://localhost:1234

# Database Configuration
DATABASE_URL=sqlite:///backend/atom_gpt.db
EOL
    echo "   ✅ Created .env file with default configuration"
    echo "   📝 Please review and update the .env file with your preferences"
else
    echo "   ✅ .env file already exists"
fi

# Initialize database
echo ""
echo "🗄️ Initializing database..."
cd backend
python -c "
import sys
sys.path.append('.')
from app import init_database
init_database()
print('✅ Database initialized successfully')
"
cd ..

# Test installation
echo ""
echo "🧪 Testing installation..."

echo "   Testing backend imports..."
cd backend
python -c "
try:
    from app import app
    print('✅ Backend imports successful')
except Exception as e:
    print(f'❌ Backend import error: {e}')
    exit(1)
" || exit 1
cd ..

echo "   Testing frontend build..."
cd frontend
if [ -d "build" ] || [ -d "dist" ]; then
    echo "✅ Frontend build successful"
else
    echo "❌ Frontend build failed"
    exit 1
fi
cd ..

# Create startup scripts
echo ""
echo "📜 Creating startup scripts..."

# Create development startup script
cat > start-dev.sh << 'EOL'
#!/bin/bash
# ATOM-GPT Development Startup Script

echo "🚀 Starting ATOM-GPT in Development Mode"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Start backend in background
echo "🔄 Starting backend server..."
source venv/bin/activate
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🔄 Starting frontend server..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ ATOM-GPT is starting up!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend:  http://localhost:8000"
echo "📊 Health:   http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user to stop
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
EOL

chmod +x start-dev.sh

# Create production startup script
cat > start-prod.sh << 'EOL'
#!/bin/bash
# ATOM-GPT Production Startup Script

echo "🚀 Starting ATOM-GPT in Production Mode"
echo "======================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start backend
echo "🔄 Starting backend server in production mode..."
cd backend
export NODE_ENV=production
python app.py
EOL

chmod +x start-prod.sh

echo ""
echo "🎉 ATOM-GPT Setup Complete!"
echo "=========================="
echo ""
echo "📋 What was installed:"
echo "   ✅ Python virtual environment with all dependencies"
echo "   ✅ Frontend built and ready"
echo "   ✅ Database initialized with default admin user"
echo "   ✅ Configuration files created"
echo "   ✅ Startup scripts created"
echo ""
echo "🚀 To start development:"
echo "   ./start-dev.sh"
echo ""
echo "🔐 Default admin account:"
echo "   Email: admin@atomgpt.local"
echo "   Password: admin123"
echo ""
echo "📖 Next steps:"
echo "   1. Review and update the .env file"
echo "   2. Run ./start-dev.sh to start development servers"
echo "   3. Visit http://localhost:3000 to access the app"
echo "   4. Check http://localhost:8000/health for backend status"
echo ""
echo "📚 For more information, see README.md"
