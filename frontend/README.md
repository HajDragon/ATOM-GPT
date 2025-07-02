# ATOM-GPT Frontend

This is the React frontend for ATOM-GPT, providing a modern web interface for interacting with the AI model.

## Features

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
- `npm run eject` - Ejects from Create React App (irreversible)

## Project Structure

```
frontend/
├── public/             # Static files
├── src/
│   ├── components/     # React components
│   │   ├── ChatInterface.tsx
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
