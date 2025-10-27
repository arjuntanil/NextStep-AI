# NextStepAI - React Frontend

## 🚀 Quick Start Guide

### Prerequisites
- Node.js 16+ and npm installed
- Python 3.9+ with backend dependencies
- Backend API running on http://127.0.0.1:8000

### Installation

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

The React app will open at `http://localhost:3000`

### Starting Both Frontend and Backend

**Option 1: Use the batch file (Windows)**
```bash
START_REACT_SYSTEM.bat
```

**Option 2: Manual start**

Terminal 1 - Backend:
```bash
START_BACKEND.bat
```

Terminal 2 - Frontend:
```bash
cd frontend
npm start
```

## 📁 Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Layout.js           # Main layout with sidebar
│   │   └── ProtectedRoute.js   # Route protection
│   ├── contexts/
│   │   └── AuthContext.js      # Authentication state
│   ├── pages/
│   │   ├── Login.js            # Login page
│   │   ├── Register.js         # Registration page
│   │   ├── Dashboard.js        # Main dashboard
│   │   ├── CVAnalyzer.js       # CV analysis feature
│   │   ├── CareerAdvisor.js    # AI career advice
│   │   ├── RAGCoach.js         # Document Q&A
│   │   ├── History.js          # User history
│   │   └── AdminDashboard.js   # Admin analytics
│   ├── services/
│   │   └── api.js              # API client
│   ├── App.js                  # Main app component
│   ├── index.js                # Entry point
│   └── index.css               # Global styles
├── package.json
└── README.md
```

## 🎨 Features

### User Features
- ✅ **User Authentication** - Login/Register with JWT
- ✅ **CV Analyzer** - Upload resume for AI analysis
- ✅ **Career Advisor** - Chat with AI for career guidance
- ✅ **RAG Coach** - Upload documents and ask questions
- ✅ **History** - View past analyses and queries
- ✅ **Responsive Design** - Works on desktop and mobile

### Admin Features
- ✅ **Admin Dashboard** - Analytics and engagement charts
- ✅ **User Statistics** - Monitor platform usage
- ✅ **Data Visualization** - Interactive charts with Recharts

## 🔧 Technology Stack

- **React 18** - UI framework
- **Material-UI (MUI)** - Component library
- **React Router** - Navigation
- **Axios** - HTTP client
- **Recharts** - Data visualization
- **Context API** - State management

## 🔐 Authentication

The app uses JWT tokens for authentication:
- Tokens stored in localStorage
- Automatic token validation on app load
- Auto-redirect to login on 401 errors
- Protected routes for authenticated users

## 🌐 API Integration

All API calls are in `src/services/api.js`:

```javascript
// Example API calls
import { cvAPI, careerAPI, authAPI } from './services/api';

// Login
const result = await authAPI.login(email, password);

// Analyze CV
const formData = new FormData();
formData.append('file', file);
const response = await cvAPI.analyzeResume(formData);

// Career advice
const response = await careerAPI.queryCareerPath(question);
```

## 🎯 Environment Configuration

The app connects to the backend at `http://127.0.0.1:8000` by default.

To change this, edit `src/services/api.js`:
```javascript
const API_BASE_URL = 'http://your-backend-url:8000';
```

## 📱 Available Routes

- `/` - Dashboard (protected)
- `/login` - Login page
- `/register` - Registration page
- `/cv-analyzer` - CV analysis tool
- `/career-advisor` - AI career chat
- `/rag-coach` - Document Q&A
- `/history` - User history
- `/admin` - Admin dashboard (admin only)

## 🚧 Development

### Running in Development Mode
```bash
npm start
```

### Building for Production
```bash
npm run build
```

### Running Tests
```bash
npm test
```

## 🔄 Migration from Streamlit

This React frontend replaces the Streamlit app (`app.py`) with a modern web application:

**Benefits:**
- ✅ Better performance and user experience
- ✅ More flexible UI customization
- ✅ Native mobile responsiveness
- ✅ Better state management
- ✅ Professional appearance
- ✅ Easier deployment options

**All Streamlit features ported to React:**
- ✅ User authentication
- ✅ CV analysis
- ✅ Career advisor chat
- ✅ RAG coach
- ✅ User history
- ✅ Admin dashboard with charts

## 🐛 Troubleshooting

### CORS Errors
Make sure the backend has CORS enabled for `http://localhost:3000`

### API Connection Failed
1. Check backend is running: `http://127.0.0.1:8000/docs`
2. Verify API_BASE_URL in `src/services/api.js`

### Dependencies Installation Failed
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Port 3000 Already in Use
```bash
# Kill process on port 3000 (Windows)
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

## 📊 Admin Dashboard

The admin dashboard shows:
- Total users, CV analyses, career queries
- User engagement over time (line chart)
- Platform health metrics
- Usage statistics

Access: Only available to users with `role: 'admin'`

## 🔒 Security Features

- JWT token authentication
- Protected routes
- Auto token validation
- Secure API communication
- Role-based access control

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review backend logs
3. Check browser console for errors
4. Ensure all dependencies are installed

## 🎉 Success!

Once started, you should see:
- React app at `http://localhost:3000`
- Backend API at `http://127.0.0.1:8000`
- API docs at `http://127.0.0.1:8000/docs`

Happy career navigating! ✨
