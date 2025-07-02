# =============================================================================
# ATOM-GPT Quick Start Script (PowerShell)
# Automatically starts both backend and frontend in separate terminals
# =============================================================================

Write-Host "🚀 Starting ATOM-GPT Project..." -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Get the current script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir

Write-Host "📁 Project Root: $ProjectRoot" -ForegroundColor Cyan
Write-Host ""

# Check if required directories exist
if (-not (Test-Path "$ProjectRoot\backend")) {
    Write-Host "❌ Backend directory not found at: $ProjectRoot\backend" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "$ProjectRoot\frontend")) {
    Write-Host "❌ Frontend directory not found at: $ProjectRoot\frontend" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Function to start a new terminal with command
function Start-TerminalWithCommand {
    param(
        [string]$Title,
        [string]$Command,
        [string]$WorkingDirectory
    )
    
    Write-Host "🖥️  Starting $Title..." -ForegroundColor Yellow
    
    # Try Windows Terminal first
    if (Get-Command wt -ErrorAction SilentlyContinue) {
        Start-Process wt -ArgumentList "new-tab", "--title", "`"$Title`"", "powershell", "-NoExit", "-Command", "cd '$WorkingDirectory'; $Command"
    }
    # Fallback to regular PowerShell
    else {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$WorkingDirectory'; Write-Host '$Title' -ForegroundColor Green; $Command"
    }
    
    Start-Sleep -Seconds 1
}

# Start backend in first terminal
Write-Host "1️⃣  Launching Backend Server (Port 8000)..." -ForegroundColor Blue
Start-TerminalWithCommand -Title "ATOM-GPT Backend" -Command "python app.py" -WorkingDirectory "$ProjectRoot\backend"

# Wait for backend to start
Write-Host "⏳ Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Start frontend in second terminal
Write-Host "2️⃣  Launching Frontend Server (Port 3000)..." -ForegroundColor Blue
Start-TerminalWithCommand -Title "ATOM-GPT Frontend" -Command "npm start" -WorkingDirectory "$ProjectRoot\frontend"

# Wait for frontend to initialize
Write-Host "⏳ Waiting for frontend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "✅ ATOM-GPT STARTUP COMPLETE!" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "🌐 Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔐 Demo Login:" -ForegroundColor Yellow
Write-Host "   Email:    admin@atomgpt.local" -ForegroundColor White
Write-Host "   Password: admin123" -ForegroundColor White
Write-Host ""
Write-Host "📋 Two terminal windows should now be open:" -ForegroundColor Magenta
Write-Host "   1. Backend terminal running Python Flask server" -ForegroundColor White
Write-Host "   2. Frontend terminal running React development server" -ForegroundColor White
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "   - Wait for both servers to fully start (green status indicators)" -ForegroundColor White
Write-Host "   - Frontend will automatically open your default browser" -ForegroundColor White
Write-Host "   - Close terminals or press Ctrl+C to stop servers" -ForegroundColor White
Write-Host ""
Write-Host "🎉 Happy coding!" -ForegroundColor Green

Read-Host "Press Enter to exit this script"
