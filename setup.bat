@echo off
REM ============================
REM ATOM-GPT Quick Setup Script (Windows)
REM ============================
REM This script sets up the ATOM-GPT project for development

echo 🚀 ATOM-GPT Quick Setup (Windows)
echo ===================================

REM Check if we're in the right directory
if not exist "README.md" (
    echo ❌ Error: Please run this script from the ATOM-GPT project root directory
    pause
    exit /b 1
)
if not exist "backend" (
    echo ❌ Error: Backend directory not found
    pause
    exit /b 1
)
if not exist "frontend" (
    echo ❌ Error: Frontend directory not found
    pause
    exit /b 1
)

REM Check required tools
echo 🔍 Checking required tools...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is required but not installed
    echo    Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is required but not installed
    echo    Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
)

npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm is required but not installed
    echo    npm usually comes with Node.js
    pause
    exit /b 1
)

echo ✅ All required tools found

REM Setup Python virtual environment
echo.
echo 🐍 Setting up Python environment...

if not exist "venv" (
    echo    Creating virtual environment...
    python -m venv venv
)

echo    Activating virtual environment...
call venv\Scripts\activate.bat

echo    Upgrading pip...
python -m pip install --upgrade pip

echo    Installing Python dependencies...
pip install -r requirements.txt

REM Setup backend additional dependencies
echo    Installing backend dependencies...
cd backend
if exist "requirements.txt" (
    pip install -r requirements.txt
)
cd ..

REM Setup frontend
echo.
echo ⚛️ Setting up Frontend...
cd frontend

if not exist "node_modules" (
    echo    Installing npm dependencies...
    npm install
) else (
    echo    Updating npm dependencies...
    npm update
)

echo    Building frontend...
npm run build

cd ..

REM Create .env file if it doesn't exist
echo.
echo 🔧 Setting up configuration...

if not exist ".env" (
    echo    Creating .env file...
    (
        echo # ATOM-GPT Configuration
        echo JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
        echo PORT=8000
        echo NODE_ENV=development
        echo.
        echo # AI Configuration
        echo USE_LOCAL_MODEL_ONLY=false
        echo MAX_TOKENS=300
        echo DEFAULT_TEMPERATURE=0.8
        echo.
        echo # LM Studio Configuration ^(optional^)
        echo LM_STUDIO_URL=http://localhost:1234
        echo.
        echo # Database Configuration
        echo DATABASE_URL=sqlite:///backend/atom_gpt.db
    ) > .env
    echo    ✅ Created .env file with default configuration
    echo    📝 Please review and update the .env file with your preferences
) else (
    echo    ✅ .env file already exists
)

REM Initialize database
echo.
echo 🗄️ Initializing database...
cd backend
python -c "import sys; sys.path.append('.'); from app import init_database; init_database(); print('✅ Database initialized successfully')"
cd ..

REM Test installation
echo.
echo 🧪 Testing installation...

echo    Testing backend imports...
cd backend
python -c "try: from app import app; print('✅ Backend imports successful'); except Exception as e: print(f'❌ Backend import error: {e}'); exit(1)"
if %errorlevel% neq 0 exit /b 1
cd ..

echo    Testing frontend build...
cd frontend
if exist "build" (
    echo ✅ Frontend build successful
) else if exist "dist" (
    echo ✅ Frontend build successful
) else (
    echo ❌ Frontend build failed
    pause
    exit /b 1
)
cd ..

REM Create startup scripts
echo.
echo 📜 Creating startup scripts...

REM Create development startup script
(
    echo @echo off
    echo REM ATOM-GPT Development Startup Script
    echo.
    echo echo 🚀 Starting ATOM-GPT in Development Mode
    echo echo ========================================
    echo.
    echo REM Check if virtual environment exists
    echo if not exist "venv" ^(
    echo     echo ❌ Virtual environment not found. Please run setup.bat first.
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo REM Start backend
    echo echo 🔄 Starting backend server...
    echo start "Backend" cmd /k "call venv\Scripts\activate.bat && cd backend && python app.py"
    echo.
    echo REM Wait a moment for backend to start
    echo timeout /t 3 /nobreak ^>nul
    echo.
    echo REM Start frontend
    echo echo 🔄 Starting frontend server...
    echo start "Frontend" cmd /k "cd frontend && npm start"
    echo.
    echo echo.
    echo echo ✅ ATOM-GPT is starting up!
    echo echo 🌐 Frontend: http://localhost:3000
    echo echo 🔧 Backend:  http://localhost:8000
    echo echo 📊 Health:   http://localhost:8000/health
    echo echo.
    echo echo Press any key to continue...
    echo pause ^>nul
) > start-dev.bat

echo.
echo 🎉 ATOM-GPT Setup Complete!
echo ==========================
echo.
echo 📋 What was installed:
echo    ✅ Python virtual environment with all dependencies
echo    ✅ Frontend built and ready
echo    ✅ Database initialized with default admin user
echo    ✅ Configuration files created
echo    ✅ Startup scripts created
echo.
echo 🚀 To start development:
echo    start-dev.bat
echo.
echo 🔐 Default admin account:
echo    Email: admin@atomgpt.local
echo    Password: admin123
echo.
echo 📖 Next steps:
echo    1. Review and update the .env file
echo    2. Run start-dev.bat to start development servers
echo    3. Visit http://localhost:3000 to access the app
echo    4. Check http://localhost:8000/health for backend status
echo.
echo 📚 For more information, see README.md
echo.
pause
