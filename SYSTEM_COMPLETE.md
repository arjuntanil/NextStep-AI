# 🚀 NextStepAI - Complete System Setup Guide

## ✅ System Status

Your NextStepAI system now includes:

- ✅ **Backend API** (FastAPI on port 8000)
- ✅ **User Application** (Streamlit on port 8501)
- ✅ **Admin Dashboard** (Streamlit on port 8502)
- ✅ **Manual Authentication** (Email/Password)
- ✅ **Admin Analytics** (Comprehensive dashboard)
- ✅ **Database Migrated** (All existing data preserved)

---

## 🎯 Quick Start (FASTEST METHOD)

### Step 1: Run the All-in-One Startup Script

```batch
START_ALL_SYSTEMS.bat
```

This single command will:
1. Start Backend API (Port 8000)
2. Start User App (Port 8501)
3. Start Admin Dashboard (Port 8502)
4. Open all URLs in your browser

---

## 🔑 Login Credentials

### Admin Account
- **Email**: `admin@gmail.com`
- **Password**: `admin`

### Existing User
- **Email**: `arjun.24pmc117@mariancollege.org`
- **Password**: `password123` (default - should be changed)

---

## 🌐 Access URLs

| System | URL | Purpose |
|--------|-----|---------|
| **User App** | http://localhost:8501 | Main application for users |
| **Admin Dashboard** | http://localhost:8502 | Admin analytics and management |
| **API Documentation** | http://localhost:8000/docs | Interactive API docs (Swagger) |

---

## 📊 Admin Dashboard Features

### Dashboard Overview
- **User Statistics**: Total users, active users (7d/30d), retention rate
- **Growth Chart**: User growth over last 30 days
- **Top Jobs**: Most recommended job titles
- **Top Skills**: Most missing skills across analyses
- **Match Distribution**: Distribution of match percentages
- **Recent Activity**: Last 20 user actions

### User Management
- **View All Users**: Paginated table with search
- **User Details**: Full history of analyses, queries, and RAG queries
- **Suspend/Activate**: Disable or enable user accounts
- **Delete Users**: GDPR-compliant data deletion
- **Create Users**: Manual user creation

### Advanced Analytics
- **Activity Heatmap**: User activity by day/hour
- **Job Distribution**: Pie chart of recommended jobs
- **Skill Trends**: Trending missing skills
- **Match Score Histogram**: Distribution of match percentages

---

## 🔧 Manual Startup (Alternative)

If you prefer to start each component individually:

### Terminal 1: Backend API
```batch
cd E:\NextStepAI
.\career_coach\Scripts\activate.bat
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: User App
```batch
cd E:\NextStepAI
.\career_coach\Scripts\activate.bat
streamlit run app.py
```

### Terminal 3: Admin Dashboard
```batch
cd E:\NextStepAI
.\career_coach\Scripts\activate.bat
streamlit run admin_dashboard.py --server.port 8502
```

---

## 🛡️ Authentication System

### User Registration (New Feature)
Users can now register manually:
1. Go to http://localhost:8501
2. Use the registration form in sidebar
3. Enter email, full name, and password
4. Get instant access token

### User Login (Replaces Google OAuth)
1. Go to http://localhost:8501
2. Enter email and password
3. Click "Login"

### Admin Login
1. Go to http://localhost:8502
2. Enter admin credentials
3. Access dashboard

---

## 📝 API Endpoints

### Authentication Endpoints
- `POST /auth/register` - Register new user
- `POST /auth/manual-login` - User login
- `POST /admin/login` - Admin login

### Admin Endpoints
- `GET /admin/stats` - Dashboard statistics
- `GET /admin/users` - List all users (paginated, searchable)
- `GET /admin/user/{id}` - User details with history
- `PUT /admin/user/{id}/suspend` - Suspend user
- `PUT /admin/user/{id}/activate` - Activate user
- `DELETE /admin/user/{id}` - Delete user and data
- `POST /admin/user/create` - Create user manually

### User Endpoints (Existing)
- `POST /analyze-resume` - Analyze resume
- `POST /ask-career-question` - Ask AI advisor
- `POST /rag-coach/ask` - RAG-based coaching

---

## 🗄️ Database Schema

### Users Table
- `id`: Primary key
- `email`: Unique email address
- `full_name`: User's full name
- `password_hash`: Bcrypt hashed password
- `role`: "user" or "admin"
- `is_active`: Boolean (for suspend/ban)
- `created_at`: Account creation timestamp
- `last_active`: Last activity timestamp

### Resume Analyses Table
- All existing fields
- `resume_filename`: Original file name
- `total_skills_count`: Skills count
- `created_at`: Analysis timestamp

### Career Queries Table
- All existing fields
- `model_used`: "finetuned" or "rag"
- `response_time_seconds`: Performance tracking
- `created_at`: Query timestamp

### RAG Coach Queries Table
- All existing fields
- `query_length`: Question word count
- `answer_length`: Response word count
- `created_at`: Query timestamp

---

## 🔍 Testing the System

### 1. Test User Registration
```python
import requests

# Register new user
response = requests.post("http://localhost:8000/auth/register", json={
    "email": "test@example.com",
    "full_name": "Test User",
    "password": "testpassword123"
})
print(response.json())
# Returns: {"access_token": "...", "token_type": "bearer"}
```

### 2. Test User Login
```python
# Login
response = requests.post("http://localhost:8000/auth/manual-login", json={
    "email": "test@example.com",
    "password": "testpassword123"
})
print(response.json())
```

### 3. Test Admin Login
```python
# Admin login
response = requests.post("http://localhost:8000/admin/login", json={
    "email": "admin@gmail.com",
    "password": "admin"
})
token = response.json()["access_token"]

# Get admin stats
headers = {"Authorization": f"Bearer {token}"}
stats = requests.get("http://localhost:8000/admin/stats", headers=headers)
print(stats.json())
```

---

## 🎨 Frontend Changes

### User App (app.py)
- **Tab 1**: "CV Analyzer" (was "Resume Analyzer")
- **Tab 2**: "AI Career Advisor" (unchanged)
- **Tab 3**: "Resume Analyzer using JD" (was "RAG Coach")
- Removed Google OAuth button
- Added manual login/registration forms

### Admin Dashboard (admin_dashboard.py)
- Complete new Streamlit application
- Login page with email/password
- 3 main pages: Dashboard, User Management, Analytics
- Interactive Plotly charts
- Real-time statistics

---

## 📦 Packages Installed

All required packages are installed in `career_coach` environment:

```
passlib[bcrypt]==1.7.4         # Password hashing
python-jose[cryptography]==3.5.0  # JWT tokens
bcrypt==4.3.0                   # Bcrypt backend
cryptography==45.0.5            # Cryptographic operations
```

---

## 🔄 Migration Summary

✅ **Database Migration Completed**
- Added 5 new columns to `users` table
- Added 3 new columns to `resume_analyses` table
- Added 3 new columns to `career_queries` table
- Added 3 new columns to `rag_coach_queries` table
- Preserved all existing data (1 user, 18 analyses, 4 queries)

✅ **Admin User Created**
- Email: admin@gmail.com
- Password: admin
- Role: admin

✅ **Existing User Migrated**
- Email: arjun.24pmc117@mariancollege.org
- Default password: password123
- All 18 resume analyses preserved
- All 4 career queries preserved

---

## 🛠️ Troubleshooting

### Backend won't start
```batch
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed
taskkill /PID <PID> /F

# Restart backend
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

### Admin dashboard won't load
```batch
# Check if port 8502 is available
netstat -ano | findstr :8502

# Manually start admin dashboard
streamlit run admin_dashboard.py --server.port 8502
```

### Can't login as admin
1. Verify admin exists:
   ```batch
   python create_admin.py
   ```
2. Check credentials:
   - Email: admin@gmail.com
   - Password: admin

### Database issues
```batch
# Re-run migration
python migrate_database.py

# Recreate admin
python create_admin.py
```

---

## 📈 Next Steps

### For Users
1. Register new account or login with existing credentials
2. Upload resume for analysis
3. Ask career questions to AI advisor
4. Use RAG coach for resume optimization

### For Admins
1. Login to admin dashboard
2. Monitor user activity and growth
3. Review job market trends
4. Manage users (suspend, activate, delete)
5. Create new users manually

---

## 🔐 Security Notes

- ⚠️ **Change default passwords**: Both admin and existing user have default passwords
- ⚠️ **JWT Secret**: Update `JWT_SECRET_KEY` in backend_api.py for production
- ⚠️ **HTTPS**: Use HTTPS in production (current setup is HTTP for localhost)
- ✅ **Password Hashing**: All passwords are bcrypt hashed
- ✅ **Role-based Access**: Admin endpoints protected by role verification
- ✅ **Token Expiry**: JWT tokens expire after 24 hours

---

## 📞 Support

If you encounter issues:
1. Check `nextstepai.db` exists
2. Verify virtual environment is activated
3. Check all ports are available (8000, 8501, 8502)
4. Review terminal output for error messages

---

## 🎉 Success Indicators

You know the system is working correctly when:
- ✅ Backend API shows "Application startup complete" on http://localhost:8000
- ✅ User app loads at http://localhost:8501 with login form
- ✅ Admin dashboard loads at http://localhost:8502 with admin login
- ✅ API docs accessible at http://localhost:8000/docs
- ✅ Admin can login and see dashboard statistics
- ✅ Users can register and login successfully

---

**Version**: 2.0 (Manual Authentication + Admin Dashboard)  
**Last Updated**: January 2025  
**Database**: nextstepai.db (SQLite)  
**Environment**: career_coach (Python virtual environment)
