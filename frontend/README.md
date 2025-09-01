# ATOM-GPT Frontend

This is the React frontend for ATOM-GPT, providing a modern web interface for interacting with the AI model. This guide will take you from zero to a fully functional web application.

## 🚀 Quick Start (0 to 100)

### Step 1: Prerequisites
```bash
# Install Node.js 16+ from nodejs.org
# Verify installation:
node --version  # Should show v16.0.0 or higher
npm --version   # Should show 8.0.0 or higher
```

### Step 2: Backend Setup First
```bash
# Make sure the backend is running first:
cd ../backend
python api/server.py
# You should see: "Server running on http://localhost:8000"
```

### Step 3: Install Frontend Dependencies
```bash
cd ../frontend  # Navigate back to frontend
npm install     # Install all dependencies (takes 1-2 minutes)
```

### Step 4: Start the Development Server
```bash
npm start
```

### Step 5: Open the Application
- Browser should auto-open to http://localhost:3000
- If not, manually open http://localhost:3000

### Step 6: Test the Interface
1. **Chat Interface**: Navigate to the main page
2. **Completion Interface**: Click "Completion" in the navigation
3. **Try Demo Account**: Click "Try Demo Account" to test authentication

🎉 **You now have a fully working ATOM-GPT web interface!**

## ✨ Features

- **Chat Interface**: Real-time conversation with ATOM-GPT
- **Completion Interface**: Text completion and sentence finishing
- **LM Studio Integration**: Enhanced responses when LM Studio is available
- **Real-time Status Monitoring**: Check model and LM Studio connectivity
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Theme**: Optimized for extended use

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- ATOM-GPT backend running (see `../backend/README.md`)

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Available Scripts

- `npm start` - Runs the app in development mode
- `npm test` - Launches the test runner
- `npm run build` - Builds the app for production
## 🎯 Detailed Usage Guide

### Chat Interface
1. **Access**: Main page (http://localhost:3000)
2. **Features**:
   - Type messages in the bottom input field
   - Press Enter or click Send to submit
   - View conversation history in the main area
   - Use sidebar to manage multiple conversations
   - Copy responses with the copy button

3. **Keyboard Shortcuts**:
   - `Ctrl + Enter`: Send message
   - `Esc`: Clear current input
   - `Ctrl + N`: New conversation

### Completion Interface
1. **Access**: Click "Completion" in navigation
2. **Features**:
   - Enter partial text for completion
   - Adjust generation settings (temperature, max tokens)
   - Use example prompts for inspiration
   - Real-time generation with progress indicators

### Authentication System
1. **Create Account**:
   - Click "Login" in the top right
   - Switch to "Register" tab
   - Fill in email and password
   - Click "Create Account"

2. **Demo Account**:
   - Click "Try Demo Account" for instant access
   - Email: admin@atomgpt.local
   - Password: admin123

3. **User Features**:
   - Persistent conversation history
   - User statistics and usage tracking
   - Profile management

### LM Studio Enhancement (Optional)
1. **Install LM Studio**: Download from lmstudio.ai
2. **Load a Model**: Choose CodeLlama, Mistral, or similar
3. **Start Server**: Enable server mode in LM Studio
4. **Configure ATOM-GPT**:
   ```bash
   # Create/edit .env file in project root:
   LM_STUDIO_URL=http://localhost:1234
   ```
5. **Enable Enhancement**: Toggle in the web app settings

## 🔧 Configuration Options

### Environment Variables
Create a `.env` file in the project root:
```env
# Backend API
REACT_APP_API_URL=http://localhost:8000

# LM Studio Integration
REACT_APP_LM_STUDIO_URL=http://localhost:1234

# Development Options
REACT_APP_DEBUG_MODE=false
```

### Customization
```javascript
// src/config/app.config.js
export const APP_CONFIG = {
  maxTokens: 100,
  temperature: 0.8,
  theme: 'dark',
  autoSave: true
};
```

## 🚨 Troubleshooting

### Common Issues

**"Cannot connect to backend"**:
```bash
# Check backend is running:
curl http://localhost:8000/health
# If not working, start backend:
cd ../backend && python api/server.py
```

**Port 3000 already in use**:
```bash
# Kill process using port 3000:
npx kill-port 3000
# Or use different port:
PORT=3001 npm start
```

**NPM install fails**:
```bash
# Clear npm cache:
npm cache clean --force
# Delete node_modules and reinstall:
rm -rf node_modules package-lock.json
npm install
```

**White screen/build errors**:
```bash
# Check for TypeScript errors:
npm run type-check
# Clear cache and rebuild:
rm -rf build
npm run build
```

### Performance Issues

**Slow response times**:
1. Check backend server logs for errors
2. Verify GPU is being used by backend
3. Reduce max_tokens in generation settings
4. Check network connectivity

**Memory usage**:
1. Close unused browser tabs
2. Restart the development server
3. Check for memory leaks in browser dev tools

## 🏗️ Development

### Project Structure
```
frontend/
├── public/                 # Static files
│   ├── index.html         # Main HTML template
│   └── manifest.json      # PWA configuration
├── src/
│   ├── components/        # React components
│   │   ├── AuthModal.tsx  # Authentication modal
│   │   ├── ChatInterface.tsx
│   │   ├── CompletionInterface.tsx
│   │   ├── StatusPanel.tsx
│   │   └── common/        # Reusable components
│   ├── contexts/          # React contexts
│   │   └── AuthContext.tsx
│   ├── services/          # API services
│   │   ├── api.ts         # Main API client
│   │   ├── auth.ts        # Authentication
│   │   └── database.ts    # Database operations
│   ├── utils/             # Utility functions
│   │   ├── chatStorage.ts # Chat persistence
│   │   └── helpers.ts     # General helpers
│   ├── styles/            # CSS/SCSS files
│   ├── App.tsx            # Main application component
│   └── index.tsx          # Application entry point
├── package.json           # Dependencies and scripts
└── tsconfig.json          # TypeScript configuration
```

### Adding New Features

1. **Create Component**:
   ```typescript
   // src/components/NewFeature.tsx
   import React from 'react';
   
   const NewFeature: React.FC = () => {
     return <div>New Feature</div>;
   };
   
   export default NewFeature;
   ```

2. **Add Route** (if needed):
   ```typescript
   // In App.tsx
   import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
   import NewFeature from './components/NewFeature';
   
   <Routes>
     <Route path="/new-feature" element={<NewFeature />} />
   </Routes>
   ```

3. **Add Navigation**:
   ```typescript
   // Add to navigation component
   <Link to="/new-feature">New Feature</Link>
   ```

### Testing

```bash
# Run all tests:
npm test

# Run specific test file:
npm test ChatInterface.test.tsx

# Run tests with coverage:
npm test -- --coverage
```

### Building for Production

```bash
# Create production build:
npm run build

# Serve production build locally:
npx serve -s build

# Deploy to server:
# Copy build/ folder contents to your web server
```

## 🌐 Deployment

### Static Hosting (Netlify, Vercel)
1. Build the project: `npm run build`
2. Upload `build/` folder to your hosting service
3. Configure environment variables in hosting dashboard

### Self-Hosted
```bash
# Install serve globally:
npm install -g serve

# Serve the build:
serve -s build -l 3000
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM node:16-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
```

## 📱 Mobile Optimization

The app is fully responsive and works on mobile devices:
- Touch-optimized interface
- Swipe gestures for navigation
- Mobile-friendly keyboard shortcuts
- Responsive grid layout

## 🎨 Customization

### Themes
```css
/* src/styles/themes.css */
:root {
  --primary-color: #your-color;
  --background-color: #your-bg;
  --text-color: #your-text;
}
```

### Component Styling
```typescript
// Use CSS modules or styled-components
import styles from './Component.module.css';
// or
import styled from 'styled-components';
```

## 📚 Related Documentation

- **[Main Project README](../README.md)** - Overall project setup
- **[Backend README](../backend/README.md)** - Backend setup and API
- **[Authentication Guide](../AUTHENTICATION_COMPLETE.md)** - Auth system details
- **[Frontend Setup Complete](SETUP_COMPLETE.md)** - Implementation status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `npm test`
6. Submit a pull request

## 📄 License

This project follows the same license as the main ATOM-GPT repository.
│   │   ├── CompletionInterface.tsx
│   │   ├── Navigation.tsx
│   │   └── StatusPanel.tsx
│   ├── services/       # API services
│   │   └── api.ts
│   ├── utils/          # Utility functions
│   ├── App.tsx         # Main app component
│   ├── index.tsx       # Entry point
│   └── index.css       # Global styles
├── package.json
└── tsconfig.json
```

## Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000
```

### Backend API

The frontend expects the ATOM-GPT backend to be running on `http://localhost:8000` by default. The backend should provide these endpoints:

- `POST /api/chat` - Send chat messages
- `POST /api/completion` - Generate text completions
- `GET /api/status` - Check model status
- `GET /api/lm-studio/status` - Check LM Studio status

## Usage

### Chat Mode

1. Navigate to the Chat tab
2. Configure settings (tokens, temperature, model)
3. Enable/disable LM Studio enhancement
4. Type your message and press Enter

### Completion Mode

1. Navigate to the Completion tab
2. Enter your prompt
3. Configure generation settings
4. Click "Generate" to get completions

### Features

- **Real-time messaging**: Responses stream in real-time
- **Enhancement indicators**: See when responses are enhanced by LM Studio
- **Responsive design**: Works on all screen sizes
- **Keyboard shortcuts**: Enter to send, Shift+Enter for new lines
- **Message history**: Previous messages are retained during the session

## API Integration

The frontend communicates with the ATOM-GPT backend through REST APIs. All requests include:

- User configuration (tokens, temperature, model selection)
- LM Studio enhancement preference
- Message history for context

Responses include:
- Generated text
- Enhancement status
- Processing metadata
- Error handling

## Customization

### Styling

The app uses Tailwind CSS for styling. You can customize:

- Colors in `tailwind.config.js`
- Global styles in `src/index.css`
- Component-specific styles inline

### Components

Each component is self-contained with TypeScript interfaces:

- `ChatInterface` - Main chat functionality
- `CompletionInterface` - Text completion
- `StatusPanel` - System status display
- `Navigation` - Route navigation

## Troubleshooting

### Common Issues

1. **Red lines in code**: Run `npm install` to install dependencies
2. **API connection errors**: Ensure backend is running on correct port
3. **Build failures**: Check TypeScript errors and fix imports

### Development Tips

- Use React Developer Tools for debugging
- Check browser console for errors
- Verify backend API responses in Network tab
- Use TypeScript strictly for better error catching

## Contributing

1. Follow TypeScript best practices
2. Add proper error handling
3. Include loading states for better UX
4. Test on multiple screen sizes
5. Update this README for new features
