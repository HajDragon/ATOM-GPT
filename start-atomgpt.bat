@echo off
REM =============================================================================
REM ATOM-GPT Quick Start Script (Windows Batch)
REM Automatically starts both backend and frontend in separate terminals
REM =============================================================================

echo 🚀 Starting ATOM-GPT Project...
echo ================================

REM Get the current script directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%"

echo 📁 Project Root: %PROJECT_ROOT%
echo.

REM Check if required directories exist
if not exist "%PROJECT_ROOT%backend" (
    echo ❌ Backend directory not found at: %PROJECT_ROOT%backend
    pause
    exit /b 1
)

if not exist "%PROJECT_ROOT%frontend" (
    echo ❌ Frontend directory not found at: %PROJECT_ROOT%frontend
    pause
    exit /b 1
)

REM Start backend in first terminal
echo 1️⃣  Launching Backend Server (Port 8000)...

REM Try Windows Terminal first, then fallback to cmd
where wt >nul 2>nul
if %errorlevel% == 0 (
    start "ATOM-GPT Backend" wt new-tab --title "ATOM-GPT Backend" cmd /k "cd /d \"%PROJECT_ROOT%backend\" && python app.py"
) else (
    start "ATOM-GPT Backend" cmd /k "cd /d \"%PROJECT_ROOT%backend\" && python app.py"
)

REM Wait for backend to start
echo ⏳ Waiting for backend to initialize...
timeout /t 3 /nobreak >nul

REM Start frontend in second terminal
echo 2️⃣  Launching Frontend Server (Port 3000)...

where wt >nul 2>nul
if %errorlevel% == 0 (
    start "ATOM-GPT Frontend" wt new-tab --title "ATOM-GPT Frontend" cmd /k "cd /d \"%PROJECT_ROOT%frontend\" && npm start"
) else (
    start "ATOM-GPT Frontend" cmd /k "cd /d \"%PROJECT_ROOT%frontend\" && npm start"
)

REM Wait for frontend to initialize
echo ⏳ Waiting for frontend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo ✅ ATOM-GPT STARTUP COMPLETE!
echo ==============================
echo.
echo 🌐 Backend:  http://localhost:8000
echo 🌐 Frontend: http://localhost:3000
echo.
echo 🔐 Demo Login:
echo    Email:    admin@atomgpt.local
echo    Password: admin123
echo.
echo 📋 Two terminal windows should now be open:
echo    1. Backend terminal running Python Flask server
echo    2. Frontend terminal running React development server
echo.
echo 💡 Tips:
echo    - Wait for both servers to fully start (green status indicators)
echo    - Frontend will automatically open your default browser
echo    - Close terminals or press Ctrl+C to stop servers
echo.
echo 🎉 Happy coding!
echo.
pause
