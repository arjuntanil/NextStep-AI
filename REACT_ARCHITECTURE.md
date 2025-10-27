# 🏗️ NextStepAI - React Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER'S BROWSER                          │
│                   http://localhost:3000                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              REACT FRONTEND (SPA)                    │  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │   Login     │  │  Register   │  │  Dashboard  │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │ CV Analyzer │  │   Career    │  │  RAG Coach  │ │  │
│  │  │             │  │   Advisor   │  │             │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐                   │  │
│  │  │   History   │  │    Admin    │                   │  │
│  │  │             │  │  Dashboard  │                   │  │
│  │  └─────────────┘  └─────────────┘                   │  │
│  │                                                       │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │         React Router (Navigation)              │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │     AuthContext (Global State)                 │ │  │
│  │  │   - User Info                                  │ │  │
│  │  │   - JWT Token                                  │ │  │
│  │  │   - Auth Methods                               │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │        API Service (Axios)                     │ │  │
│  │  │   - HTTP Client                                │ │  │
│  │  │   - Token Injection                            │ │  │
│  │  │   - Error Handling                             │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↕️
                      HTTP/HTTPS
                  (CORS Enabled)
                            ↕️
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                          │
│                 http://127.0.0.1:8000                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              API ENDPOINTS                           │  │
│  │                                                       │  │
│  │  Auth:                                               │  │
│  │  POST /auth/register                                 │  │
│  │  POST /auth/manual-login                             │  │
│  │  GET  /users/me                                      │  │
│  │                                                       │  │
│  │  CV Analysis:                                        │  │
│  │  POST /analyze_resume/                               │  │
│  │                                                       │  │
│  │  Career Advisor:                                     │  │
│  │  POST /query-career-path/                            │  │
│  │  POST /career-advice-ai                              │  │
│  │  GET  /model-status                                  │  │
│  │                                                       │  │
│  │  RAG Coach:                                          │  │
│  │  POST /rag-coach/upload                              │  │
│  │  POST /rag-coach/query                               │  │
│  │  GET  /rag-coach/status                              │  │
│  │                                                       │  │
│  │  History:                                            │  │
│  │  GET  /history/analyses                              │  │
│  │  GET  /history/queries                               │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              AI MODELS                               │  │
│  │                                                       │  │
│  │  - Fine-tuned GPT-2 (Career Advice)                 │  │
│  │  - Google Gemini (CV Analysis)                      │  │
│  │  - FAISS + RAG (Document Q&A)                       │  │
│  │  - TF-IDF Classifier (Job Matching)                 │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              DATABASE (SQLite)                       │  │
│  │                                                       │  │
│  │  Tables:                                             │  │
│  │  - users (authentication)                            │  │
│  │  - resume_analyses (CV history)                      │  │
│  │  - career_queries (advice history)                   │  │
│  │  - rag_coach_queries (RAG history)                   │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Diagrams

### 1. Authentication Flow

```
┌─────────┐         ┌─────────┐         ┌─────────┐
│  User   │────────▶│ React   │────────▶│ Backend │
│ Browser │         │  App    │         │   API   │
└─────────┘         └─────────┘         └─────────┘
     │                   │                    │
     │  Enter Email/PW   │                    │
     │──────────────────▶│                    │
     │                   │  POST /auth/login  │
     │                   │───────────────────▶│
     │                   │                    │
     │                   │   JWT Token        │
     │                   │◀───────────────────│
     │                   │                    │
     │                   │ Store in           │
     │                   │ localStorage       │
     │                   │                    │
     │  Redirect to /    │                    │
     │◀──────────────────│                    │
     │                   │                    │
```

### 2. CV Analysis Flow

```
┌─────────┐         ┌─────────┐         ┌─────────┐
│  User   │────────▶│ CV Page │────────▶│ Backend │
│         │         │         │         │         │
└─────────┘         └─────────┘         └─────────┘
     │                   │                    │
     │  Upload File      │                    │
     │──────────────────▶│                    │
     │                   │                    │
     │  Click Analyze    │                    │
     │──────────────────▶│                    │
     │                   │  FormData +        │
     │                   │  JWT Token         │
     │                   │───────────────────▶│
     │                   │                    │
     │                   │                    ├──┐
     │                   │                    │  │ Extract Skills
     │                   │                    │  │ Match Jobs
     │                   │                    │  │ Save to DB
     │                   │                    │◀─┘
     │                   │                    │
     │                   │  Results JSON      │
     │                   │◀───────────────────│
     │                   │                    │
     │  Display Results  │                    │
     │◀──────────────────│                    │
     │                   │                    │
```

### 3. Career Advisor Flow

```
┌─────────┐         ┌─────────┐         ┌─────────┐
│  User   │────────▶│ Career  │────────▶│ Backend │
│         │         │  Page   │         │         │
└─────────┘         └─────────┘         └─────────┘
     │                   │                    │
     │  Type Question    │                    │
     │──────────────────▶│                    │
     │                   │                    │
     │  Click Send       │                    │
     │──────────────────▶│                    │
     │                   │  POST /query-      │
     │                   │  career-path/      │
     │                   │───────────────────▶│
     │                   │                    │
     │                   │                    ├──┐
     │                   │                    │  │ Fine-tuned
     │                   │                    │  │ GPT-2 Model
     │                   │                    │  │ Generate
     │                   │                    │◀─┘
     │                   │                    │
     │                   │  AI Response       │
     │                   │◀───────────────────│
     │                   │                    │
     │  Show in Chat     │                    │
     │◀──────────────────│                    │
     │                   │                    │
```

## Component Hierarchy

```
App.js
├── AuthProvider (Context)
│   └── Router
│       ├── Login (Public)
│       ├── Register (Public)
│       └── ProtectedRoute
│           └── Layout
│               ├── AppBar
│               ├── Drawer (Sidebar)
│               └── Outlet (Page Content)
│                   ├── Dashboard
│                   ├── CVAnalyzer
│                   ├── CareerAdvisor
│                   ├── RAGCoach
│                   ├── History
│                   └── AdminDashboard
```

## State Management

```
┌──────────────────────────────────────┐
│         AuthContext (Global)         │
├──────────────────────────────────────┤
│  State:                              │
│  - user: { email, role, ... }        │
│  - token: "JWT_TOKEN_HERE"           │
│  - loading: boolean                  │
│                                      │
│  Methods:                            │
│  - login(email, password)            │
│  - register(email, password, name)   │
│  - logout()                          │
│  - isAuthenticated()                 │
│  - isAdmin()                         │
└──────────────────────────────────────┘
         ↕️ (useContext)
┌──────────────────────────────────────┐
│        Page Components               │
├──────────────────────────────────────┤
│  Local State (useState):             │
│  - form inputs                       │
│  - loading states                    │
│  - error messages                    │
│  - API response data                 │
│  - UI state (modals, etc.)           │
└──────────────────────────────────────┘
```

## API Communication Pattern

```
Component
   ↓
useState (local state)
   ↓
Event Handler (onClick, onSubmit)
   ↓
API Service Method
   ↓
Axios Instance
   ↓
Request Interceptor → Add JWT Token
   ↓
HTTP Request to Backend
   ↓
Backend Processing
   ↓
HTTP Response
   ↓
Response Interceptor → Check 401
   ↓
Component Callback
   ↓
setState (update state)
   ↓
Component Re-renders
   ↓
UI Updates
```

## Technology Stack Layers

```
┌────────────────────────────────────────────┐
│         Presentation Layer                 │
│  - React Components                        │
│  - Material-UI                             │
│  - CSS-in-JS                               │
│  - Recharts                                │
└────────────────────────────────────────────┘
                   ↕️
┌────────────────────────────────────────────┐
│         Application Layer                  │
│  - React Router (Navigation)               │
│  - Context API (State)                     │
│  - Custom Hooks                            │
│  - Event Handlers                          │
└────────────────────────────────────────────┘
                   ↕️
┌────────────────────────────────────────────┐
│         Data Layer                         │
│  - Axios (HTTP Client)                     │
│  - API Services                            │
│  - Local Storage                           │
│  - Form State                              │
└────────────────────────────────────────────┘
                   ↕️
┌────────────────────────────────────────────┐
│         Backend Layer                      │
│  - FastAPI                                 │
│  - JWT Authentication                      │
│  - SQLAlchemy ORM                          │
│  - AI Models                               │
└────────────────────────────────────────────┘
```

## Routing Structure

```
/
├── /login              → Login.js (public)
├── /register           → Register.js (public)
└── / (protected)       → Layout.js
    ├── /               → Dashboard.js
    ├── /cv-analyzer    → CVAnalyzer.js
    ├── /career-advisor → CareerAdvisor.js
    ├── /rag-coach      → RAGCoach.js
    ├── /history        → History.js
    └── /admin          → AdminDashboard.js (admin only)
```

## Build Process

```
Source Code (src/)
       ↓
  React Scripts
       ↓
   Webpack
       ↓
  Babel (JSX → JS)
       ↓
  Minification
       ↓
  Code Splitting
       ↓
  Build Folder
       ↓
Production Bundle
  (index.html + JS/CSS chunks)
```

## Deployment Architecture

```
┌──────────────────────────────────────┐
│        CDN / Static Hosting          │
│   (Netlify, Vercel, S3, etc.)        │
│                                      │
│  - Serves React build files          │
│  - HTTPS enabled                     │
│  - Global distribution               │
└──────────────────────────────────────┘
              ↕️ HTTPS
┌──────────────────────────────────────┐
│         User's Browser               │
│   (React App Running)                │
└──────────────────────────────────────┘
              ↕️ API Calls
┌──────────────────────────────────────┐
│        Backend Server                │
│   (Heroku, AWS, Railway, etc.)       │
│                                      │
│  - FastAPI                           │
│  - Database                          │
│  - AI Models                         │
└──────────────────────────────────────┘
```

## Security Flow

```
┌────────────────────────────────────────────┐
│  1. User logs in                           │
│     ↓                                      │
│  2. Backend creates JWT token              │
│     ↓                                      │
│  3. Token stored in localStorage           │
│     ↓                                      │
│  4. Axios interceptor adds token           │
│     to all requests                        │
│     ↓                                      │
│  5. Backend validates token                │
│     ↓                                      │
│  6. If valid → Process request             │
│     If invalid → Return 401                │
│     ↓                                      │
│  7. Response interceptor catches 401       │
│     ↓                                      │
│  8. Auto-redirect to login                 │
└────────────────────────────────────────────┘
```

## Summary

This architecture provides:

✅ **Separation of Concerns** - Frontend/Backend decoupled
✅ **Scalability** - Independent scaling of components
✅ **Security** - JWT authentication, CORS, protected routes
✅ **Maintainability** - Clear component structure
✅ **Performance** - Optimized builds, code splitting
✅ **User Experience** - SPA, smooth navigation
✅ **Developer Experience** - Hot reload, clear structure

---

*NextStepAI React Architecture - Production Ready* 🚀
