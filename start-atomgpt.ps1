# =============================================================================
# ATOM-GPT Quick Start Script (PowerShell)
# Automatically starts backend and frontend with LM Studio checks
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

# Check LM Studio connection
Write-Host "🔍 Checking LM Studio connection..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/v1/models" -TimeoutSec 3 -ErrorAction Stop
    Write-Host "✅ LM Studio detected and running on port 8080" -ForegroundColor Green
} catch {
    Write-Host "⚠️  LM Studio not detected on port 8080" -ForegroundColor Yellow
    Write-Host "Please start LM Studio first and load a model" -ForegroundColor Yellow
    $continue = Read-Host "Press Enter to continue anyway or Ctrl+C to exit"
}

# Kill any existing backend processes
Write-Host "🛑 Stopping any existing backend processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -eq "python" -and $_.MainWindowTitle -like "*ATOM-GPT Backend*"} | Stop-Process -Force -ErrorAction SilentlyContinue

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

# Wait for backend to start and test connection
Write-Host "⏳ Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

# Check if backend is responding
$backendReady = $false
$maxAttempts = 10
$attempt = 0

while (-not $backendReady -and $attempt -lt $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/status" -TimeoutSec 2 -ErrorAction Stop
        $backendReady = $true
        Write-Host "✅ Backend is ready and responding" -ForegroundColor Green
    } catch {
        $attempt++
        Write-Host "⏳ Backend still starting, waiting... (attempt $attempt/$maxAttempts)" -ForegroundColor Yellow
        Start-Sleep -Seconds 2
    }
}

if (-not $backendReady) {
    Write-Host "⚠️  Backend may not be responding, but continuing..." -ForegroundColor Yellow
}

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
Write-Host "🤖 LM Studio: http://localhost:8080" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔐 Demo Login:" -ForegroundColor Yellow
Write-Host "   Email:    admin@atomgpt.local" -ForegroundColor White
Write-Host "   Password: admin123" -ForegroundColor White
Write-Host ""
Write-Host "📋 Three services should now be running:" -ForegroundColor Magenta
Write-Host "   1. LM Studio (external) - AI model server" -ForegroundColor White
Write-Host "   2. Backend terminal - Python Flask API server" -ForegroundColor White
Write-Host "   3. Frontend terminal - React development server" -ForegroundColor White
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "   - Wait for all servers to fully start (green status indicators)" -ForegroundColor White
Write-Host "   - Frontend will automatically open your default browser" -ForegroundColor White
Write-Host "   - Close terminals or press Ctrl+C to stop servers" -ForegroundColor White
Write-Host "   - Make sure LM Studio has a model loaded for AI features" -ForegroundColor White
Write-Host ""
Write-Host "🎉 Happy coding!" -ForegroundColor Green

Read-Host "Press Enter to exit this script"
