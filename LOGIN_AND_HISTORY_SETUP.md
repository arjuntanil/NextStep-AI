# 🔐 User Authentication & History Storage - Complete Guide

## ✅ What's Implemented

Your NextStepAI project now has **full user authentication and history storage** capabilities! Here's what works:

### 1. **Google OAuth Login**
- ✅ Login with Google account (OAuth 2.0)
- ✅ JWT token-based session management
- ✅ Automatic token validation and refresh
- ✅ Secure logout functionality

### 2. **Automatic History Storage**
All user activities are automatically saved when logged in:
- ✅ **Resume Analyzer** - Saves job recommendations, skill matches, and skill gaps
- ✅ **AI Career Advisor** - Saves all career path queries and matched job groups
- ✅ **RAG Coach** - Saves questions, AI-generated answers, and source documents

### 3. **History Viewing**
- ✅ "My History" tab shows all past analyses and queries
- ✅ Expandable cards for detailed view
- ✅ Source document tracking for RAG Coach queries
- ✅ One-click refresh button to reload history

---

## 🚀 How to Enable Login

### Step 1: Get Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable **Google+ API**
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
5. Configure OAuth consent screen:
   - Application type: **Web application**
   - Authorized redirect URIs: `http://localhost:8000/auth/callback`
6. Copy your **Client ID** and **Client Secret**

### Step 2: Configure Environment Variables

Edit your `.env` file:

```env
# Required for login functionality
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# Generate a random JWT secret (run this in Python):
# python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=your_random_jwt_secret_key_here

# Required for AI features
GOOGLE_API_KEY=your_gemini_api_key_here

# Frontend URL (default is correct for local development)
STREAMLIT_FRONTEND_URL=http://localhost:8501
```

### Step 3: Initialize Database

The database will be automatically created when you start the backend. Make sure the `nextstepai.db` file gets created in your project root.

### Step 4: Start the Application

```powershell
# Terminal 1: Start Backend
cd E:\NextStepAI
.\RESTART_BACKEND.bat

# Terminal 2: Start Frontend
cd E:\NextStepAI
.\START_FRONTEND.bat
```

---

## 📊 Database Schema

Your project now has these database tables:

### **users** table
- `id` - Primary key
- `email` - User's Google email (unique)
- `full_name` - User's full name from Google

### **resume_analyses** table
- `id` - Primary key
- `owner_id` - Foreign key to users
- `recommended_job_title` - AI-recommended job
- `match_percentage` - Skill match score (0-100)
- `skills_to_add` - JSON array of missing skills

### **career_queries** table
- `id` - Primary key
- `owner_id` - Foreign key to users
- `user_query_text` - User's career question
- `matched_job_group` - Matched job category

### **rag_coach_queries** table (NEW!)
- `id` - Primary key
- `owner_id` - Foreign key to users
- `question` - User's RAG Coach question
- `answer` - AI-generated answer
- `sources` - JSON array of source documents

---

## 🎯 User Flow

### **Without Login**
1. User can use all features normally
2. ⚠️ **No data is saved** - results are lost after session ends
3. Sidebar shows: "🔐 **Login to save and view your history**"
4. "My History" tab is hidden

### **With Login**
1. User clicks **"🔑 Login with Google"** in sidebar
2. Redirected to Google OAuth consent screen
3. After authorization, redirected back with JWT token
4. Sidebar shows: "✅ Logged in as user@gmail.com"
5. All activities automatically saved to database:
   - Resume analysis → `resume_analyses` table
   - Career queries → `career_queries` table
   - RAG Coach queries → `rag_coach_queries` table
6. "My History" tab appears with all past data
7. User can click **"🔄 Refresh History"** to reload latest data
8. Click **"🚪 Logout"** to end session

---

## 🔍 What Gets Stored

### Resume Analyzer Storage
When a logged-in user analyzes a resume:
```json
{
  "owner_id": 1,
  "recommended_job_title": "Full Stack Developer",
  "match_percentage": 85,
  "skills_to_add": ["Azure", "CI/CD", "Kubernetes", "GraphQL"]
}
```

### AI Career Advisor Storage
When a logged-in user asks a career question:
```json
{
  "owner_id": 1,
  "user_query_text": "Tell me about a career in Data Science",
  "matched_job_group": "Data Scientist"
}
```

### RAG Coach Storage (NEW!)
When a logged-in user asks a RAG Coach question:
```json
{
  "owner_id": 1,
  "question": "What skills should I add based on the job description?",
  "answer": "Based on the job description, you should focus on adding...",
  "sources": ["resume.pdf", "job_description.pdf", "career_guides/data_science.pdf"]
}
```

---

## 🛠️ Backend Implementation Details

### Authentication Flow
```python
# backend_api.py (lines 470-495)

@app.get("/auth/login")
async def auth_login():
    """Initiate Google OAuth login"""
    return google_sso.get_login_redirect()

@app.get("/auth/callback")
async def auth_callback(request: Request):
    """Handle OAuth callback and issue JWT token"""
    # 1. Verify OAuth code with Google
    # 2. Get user info (email, name)
    # 3. Create/update user in database
    # 4. Generate JWT token
    # 5. Redirect to frontend with token
```

### History Storage Logic
```python
# Resume Analyzer (line 759)
if current_user:
    new_analysis = ResumeAnalysis(
        owner_id=current_user.id,
        recommended_job_title=recommended_job_title,
        match_percentage=int(match_percentage),
        skills_to_add=json.dumps(skills_to_add)
    )
    db.add(new_analysis)
    db.commit()

# Career Advisor (line 836)
if current_user:
    new_query = CareerQuery(
        owner_id=current_user.id,
        user_query_text=query.text,
        matched_job_group=matched_job_group
    )
    db.add(new_query)
    db.commit()

# RAG Coach (NEW - lines 1804-1812, 1826-1834)
if current_user:
    new_rag_query = RAGCoachQuery(
        owner_id=current_user.id,
        question=query.question,
        answer=result['answer'],
        sources=json.dumps(sources)
    )
    db.add(new_rag_query)
    db.commit()
```

### History Retrieval Endpoints
```python
# Get resume analyses
GET /history/analyses
# Returns: [{ id, recommended_job_title, match_percentage, skills_to_add }, ...]

# Get career queries
GET /history/queries
# Returns: [{ id, user_query_text, matched_job_group }, ...]

# Get RAG Coach queries (NEW!)
GET /history/rag-queries
# Returns: [{ id, question, answer, sources }, ...]
```

---

## 🎨 Frontend UI Updates

### Sidebar Changes
```python
# app.py (lines 58-69)

if st.session_state.token:
    st.sidebar.success(f"✅ Logged in as {st.session_state.user_info.get('email')}")
    if st.sidebar.button("🚪 Logout"):
        # Clear session and redirect
else:
    st.sidebar.info("🔐 **Login to save and view your history**")
    if st.sidebar.button("🔑 Login with Google", type="primary"):
        # Redirect to OAuth endpoint
    st.sidebar.caption("Without login, results won't be saved.")
```

### History Tab
```python
# app.py (lines 454-510)

with tabs[3]:  # "My History" tab (only visible when logged in)
    if st.button("🔄 Refresh History"):
        # Fetch all history from backend
    
    # Display Resume Analyses
    st.subheader("📄 Past Resume Analyses")
    for item in analyses:
        st.expander(f"{item['recommended_job_title']} ({item['match_percentage']}%)")
    
    # Display Career Queries
    st.subheader("💬 Past AI Career Advisor Queries")
    for item in queries:
        st.info(f"Asked about '{item['user_query_text']}'")
    
    # Display RAG Coach Queries (NEW!)
    st.subheader("🧑‍💼 Past RAG Coach Interactions")
    for item in rag_queries:
        with st.expander(f"Q: {item['question']}"):
            st.markdown("**Answer:**")
            st.write(item['answer'])
            st.caption(f"📄 Sources: {', '.join(json.loads(item['sources']))}")
```

---

## 🧪 Testing the Login Feature

### Manual Testing Steps

1. **Start the application:**
   ```powershell
   # Terminal 1: Backend
   cd E:\NextStepAI
   python -m uvicorn backend_api:app --reload --port 8000
   
   # Terminal 2: Frontend
   cd E:\NextStepAI
   streamlit run app.py
   ```

2. **Test login flow:**
   - Open http://localhost:8501
   - Click "🔑 Login with Google" in sidebar
   - Authorize with your Google account
   - Should redirect back with green "✅ Logged in as..." message

3. **Test history storage:**
   - Upload a resume → Analyze
   - Go to "My History" tab
   - Click "🔄 Refresh History"
   - Should see your analysis

4. **Test RAG Coach history:**
   - Upload resume + job description PDFs
   - Ask a question in RAG Coach
   - Check "My History" → "Past RAG Coach Interactions"

---

## 🔒 Security Features

### ✅ Implemented Security
- **Environment-based secrets** - No hardcoded credentials
- **JWT tokens** - Stateless authentication with expiration
- **OAuth 2.0** - Secure Google login flow
- **HTTPS redirect URLs** - Production-ready OAuth callbacks
- **Password-free** - No password storage/hashing needed
- **Foreign key constraints** - Database referential integrity

### 🛡️ Production Recommendations
```env
# Use strong, randomly generated secrets
JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# Use HTTPS in production
STREAMLIT_FRONTEND_URL=https://yourdomain.com

# Update OAuth redirect URI
GOOGLE_REDIRECT_URI=https://yourdomain.com/auth/callback
```

---

## 📋 API Endpoints Summary

### Authentication
- `GET /auth/login` - Initiate Google OAuth login
- `GET /auth/callback` - Handle OAuth callback, issue JWT token
- `GET /users/me` - Get current user info (requires JWT)

### Resume Analysis
- `POST /analyze_resume/` - Analyze resume (auto-saves if logged in)

### Career Advisor
- `POST /query-career-path/` - Ask career question (auto-saves if logged in)

### RAG Coach
- `POST /rag-coach/upload` - Upload PDFs for RAG Coach
- `POST /rag-coach/query` - Ask RAG Coach question (auto-saves if logged in)

### History Retrieval
- `GET /history/analyses` - Get user's resume analyses (requires login)
- `GET /history/queries` - Get user's career queries (requires login)
- `GET /history/rag-queries` - Get user's RAG Coach queries (requires login) ✨ NEW

---

## 🐛 Troubleshooting

### "Login button doesn't redirect"
**Solution:** Check if `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are set in `.env`

### "Token expired" error
**Solution:** Logout and login again. JWT tokens expire after a set duration (configurable)

### "History not showing"
**Solution:** 
1. Make sure you're logged in
2. Click "🔄 Refresh History" button
3. Check if `nextstepai.db` file exists in project root

### "Database error: no such table"
**Solution:** Delete `nextstepai.db` file and restart backend to recreate tables with new schema

### "OAuth redirect mismatch"
**Solution:** Make sure the redirect URI in Google Cloud Console matches exactly:
- Development: `http://localhost:8000/auth/callback`
- Production: `https://yourdomain.com/auth/callback`

---

## 🎯 Key Benefits

### For Users
✅ **Persistent history** - Never lose your analysis results
✅ **Cross-device access** - Login from any device to see your data
✅ **Progress tracking** - See how your skills improve over time
✅ **No manual saving** - Everything saves automatically when logged in

### For Development
✅ **User analytics** - Track which features are most used
✅ **Personalization** - Build user-specific recommendations
✅ **Scalability** - Ready for multi-user deployment
✅ **Security** - Industry-standard OAuth authentication

---

## 📝 Changes Summary

### Files Modified
1. **`models.py`** - Added `RAGCoachQuery` model with user relationship
2. **`backend_api.py`** - Added history saving to RAG Coach query endpoint
3. **`app.py`** - Added login button, RAG history display, improved UI

### Database Changes
- New table: `rag_coach_queries` (id, owner_id, question, answer, sources)
- New relationship: `User.rag_queries` → `RAGCoachQuery.owner`

### New Features
- 🔑 Visible login button in sidebar
- 📚 RAG Coach history storage and display
- 🔄 Refresh button for history
- 📊 Improved history tab with source tracking

---

## ✨ What's Next?

### Optional Enhancements
1. **Export History** - Download history as PDF/CSV
2. **Delete History** - Allow users to delete specific entries
3. **Search History** - Filter by date, job title, or keywords
4. **Comparison View** - Compare multiple resume analyses side-by-side
5. **Email Notifications** - Send weekly progress reports
6. **Social Login** - Add GitHub, LinkedIn OAuth options

---

## 🚀 Ready to Use!

Your login and history storage system is **fully implemented and ready to use**. Just follow the setup steps above to configure Google OAuth credentials, and you're good to go!

**Questions?** Check the troubleshooting section or review the code comments in `backend_api.py` and `app.py`.
