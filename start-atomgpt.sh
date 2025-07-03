#!/bin/bash
# =============================================================================
# ATOM-GPT Quick Start Script
# Automatically starts both backend and frontend in separate terminals
# =============================================================================

echo "🚀 Starting ATOM-GPT Project..."
echo "================================"

# Get the current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "📁 Project Root: $PROJECT_ROOT"
echo ""

# Function to detect terminal emulator and start commands
start_terminal_with_command() {
    local title="$1"
    local command="$2"
    local working_dir="$3"
    
    echo "🖥️  Starting $title..."
    
    # For Windows Git Bash / MSYS2
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Use Windows Terminal if available
        if command -v wt &> /dev/null; then
            wt new-tab --title "$title" cmd /c "cd /d \"$working_dir\" && $command"
        # Use cmd directly
        elif command -v cmd &> /dev/null; then
            cmd //c "start \"$title\" cmd /k \"cd /d $working_dir && $command\""
        # Fallback to mintty (Git Bash default)
        else
            mintty -t "$title" -e bash -c "cd '$working_dir' && $command; exec bash"
        fi
    
    # For WSL (Windows Subsystem for Linux)
    elif [[ -n "$WSL_DISTRO_NAME" ]]; then
        # Use Windows Terminal with WSL
        if command -v wt.exe &> /dev/null; then
            wt.exe new-tab --title "$title" bash -c "cd '$working_dir' && $command"
        else
            # Fallback to cmd with WSL
            cmd.exe /c "start \"$title\" bash -c \"cd '$working_dir' && $command; exec bash\""
        fi
    
    # For Linux
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v gnome-terminal &> /dev/null; then
            gnome-terminal --title="$title" --working-directory="$working_dir" -- bash -c "$command; exec bash"
        elif command -v konsole &> /dev/null; then
            konsole --new-tab --workdir "$working_dir" -e bash -c "$command; exec bash"
        elif command -v xterm &> /dev/null; then
            xterm -title "$title" -e "cd '$working_dir' && $command; exec bash" &
        else
            echo "❌ No supported terminal found for Linux"
            return 1
        fi
    
    # For macOS
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        osascript -e "tell application \"Terminal\" to do script \"cd '$working_dir' && $command\""
    
    else
        echo "❌ Unsupported operating system: $OSTYPE"
        return 1
    fi
    
    sleep 1  # Small delay between terminal launches
}

# Check if required directories exist
if [[ ! -d "$PROJECT_ROOT/backend" ]]; then
    echo "❌ Backend directory not found at: $PROJECT_ROOT/backend"
    exit 1
fi

if [[ ! -d "$PROJECT_ROOT/frontend" ]]; then
    echo "❌ Frontend directory not found at: $PROJECT_ROOT/frontend"
    exit 1
fi

# Start backend in first terminal
echo "1️⃣  Launching Backend Server (Port 8000)..."
start_terminal_with_command "ATOM-GPT Backend" "python app.py" "$PROJECT_ROOT/backend"

# Wait a moment for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 3

# Start frontend in second terminal  
echo "2️⃣  Launching Frontend Server (Port 3000)..."
start_terminal_with_command "ATOM-GPT Frontend" "npm start" "$PROJECT_ROOT/frontend"

# Wait a moment and then show status
echo "⏳ Waiting for frontend to initialize..."
sleep 5

echo ""
echo "✅ ATOM-GPT STARTUP COMPLETE!"
echo "=============================="
echo ""
echo "🌐 Backend:  http://localhost:8000"
echo "🌐 Frontend: http://localhost:3000"
echo ""
echo "🔐 Demo Login:"
echo "   Email:    admin@atomgpt.local"
echo "   Password: admin123"
echo ""
echo "📋 Two terminal windows should now be open:"
echo "   1. Backend terminal running Python Flask server"
echo "   2. Frontend terminal running React development server"
echo ""
echo "💡 Tips:"
echo "   - Wait for both servers to fully start (green status indicators)"
echo "   - Frontend will automatically open your default browser"
echo "   - Close terminals or press Ctrl+C to stop servers"
echo ""
echo "🎉 Happy coding!"
