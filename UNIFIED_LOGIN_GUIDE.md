# 🚀 NextStepAI - Unified Login System

## ✨ What's New?

You now have a **unified login portal** that automatically redirects users based on their role!

- **Admin users** → Automatically redirected to Admin Dashboard
- **Regular users** → Automatically redirected to User App

## 🎯 Quick Start

### Option 1: Start Complete System (Recommended)

```bash
.\START_COMPLETE_SYSTEM.bat
```

This starts all 4 services:
1. **Backend API** (port 8000)
2. **Login Portal** (port 8500) ⭐ **START HERE!**
3. **User App** (port 8501)
4. **Admin Dashboard** (port 8502)

### Option 2: Start Services Individually

```bash
# 1. Start Backend (Required)
.\START_BACKEND.bat

# 2. Start Login Portal
.\START_LOGIN_PORTAL.bat

# 3. Start User App (Optional - for direct access)
.\START_FRONTEND.bat

# 4. Start Admin Dashboard (Optional - for direct access)
.\START_ADMIN_DASHBOARD.bat
```

## 🔐 How to Login

### Step 1: Open Login Portal
Go to: **http://localhost:8500**

### Step 2: Enter Credentials

**For Admin Access:**
- Email: `admin@gmail.com`
- Password: `admin`
- ✅ Will redirect to → http://localhost:8502 (Admin Dashboard)

**For User Access:**
- Email: Your registered email
- Password: Your password
- ✅ Will redirect to → http://localhost:8501 (User App)

### Step 3: Automatic Redirect
The system will automatically detect your role and redirect you to the appropriate interface!

## 🌐 System URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Login Portal** | http://localhost:8500 | ⭐ **Main Entry Point** - Universal login |
| User App | http://localhost:8501 | Career navigation for users |
| Admin Dashboard | http://localhost:8502 | Analytics & user management |
| Backend API | http://localhost:8000 | REST API backend |
| API Docs | http://localhost:8000/docs | Interactive API documentation |

## 🎨 Features

### Login Portal (Port 8500)
- ✅ Beautiful gradient UI
- ✅ Role-based automatic redirection
- ✅ Token-based authentication
- ✅ Error handling and validation
- ✅ Direct links to both dashboards

### Admin Dashboard (Port 8502)
- ✅ Real-time analytics
- ✅ User growth charts
- ✅ Top jobs and skills tracking
- ✅ User management
- ✅ Activity monitoring
- ✅ Auto-login from portal

### User App (Port 8501)
- ✅ CV Analysis
- ✅ AI Career Advisor
- ✅ Resume Analyzer with JD
- ✅ History tracking
- ✅ Auto-login from portal

## 🔧 Technical Details

### How It Works

1. **User enters credentials** in Login Portal (port 8500)
2. **Portal authenticates** via Backend API
3. **Backend returns** user role (admin/user)
4. **Portal generates** secure redirect link with token
5. **User clicks link** → Auto-redirected to appropriate dashboard
6. **Dashboard validates token** and logs user in automatically

### Token Flow

```
Login Portal (8500)
    ↓ (authenticate)
Backend API (8000)
    ↓ (return token + role)
Login Portal (8500)
    ↓ (redirect with token in URL)
Admin Dashboard (8502) OR User App (8501)
    ↓ (validate token)
Auto-login successful!
```

### Security Features

- ✅ JWT token-based authentication
- ✅ 24-hour token expiry
- ✅ Bcrypt password hashing
- ✅ Token cleared from URL after auto-login
- ✅ HTTPS-ready architecture

## 📝 User Workflow Examples

### Admin Workflow
```
1. Visit http://localhost:8500
2. Login: admin@gmail.com / admin
3. Click "Open Admin Dashboard" button
4. ✅ Auto-logged into Admin Dashboard
5. View analytics, manage users, monitor system
```

### User Workflow
```
1. Visit http://localhost:8500
2. Login with your credentials
3. Click "Open Career Navigator" button
4. ✅ Auto-logged into User App
5. Upload resume, get career advice, view history
```

### New User Workflow
```
1. Visit http://localhost:8500
2. Click "Register in the User App" link
3. Register at http://localhost:8501
4. Return to http://localhost:8500
5. Login and get auto-redirected
```

## 🛠️ Troubleshooting

### "Cannot connect to backend server"
**Solution:** Start the backend first
```bash
.\START_BACKEND.bat
```

### "Invalid credentials"
**Solution:** 
- Admin: Use `admin@gmail.com` / `admin`
- User: Register first at http://localhost:8501

### Port Already in Use
**Solution:** Close existing instances
```bash
# Kill all Python/Streamlit processes
taskkill /F /IM python.exe
taskkill /F /IM streamlit.exe
```

### Auto-redirect not working
**Solution:** 
- Click the manual redirect button
- Browser may block automatic redirects
- Try a different browser

## 📊 System Architecture

```
┌─────────────────────────────────────────────┐
│         Login Portal (Port 8500)            │
│         ⭐ Main Entry Point                 │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌──────────────────┐
│ Admin Panel   │   │   User App       │
│ (Port 8502)   │   │   (Port 8501)    │
│               │   │                  │
│ - Analytics   │   │ - CV Analysis    │
│ - Users Mgmt  │   │ - AI Advisor     │
│ - Monitoring  │   │ - JD Matching    │
└───────┬───────┘   └────────┬─────────┘
        │                    │
        └──────────┬─────────┘
                   ▼
        ┌──────────────────────┐
        │  Backend API         │
        │  (Port 8000)         │
        │                      │
        │  - Authentication    │
        │  - Database          │
        │  - Business Logic    │
        └──────────────────────┘
```

## 🎯 Next Steps

1. **Start the system:**
   ```bash
   .\START_COMPLETE_SYSTEM.bat
   ```

2. **Test admin login:**
   - Go to http://localhost:8500
   - Login as admin
   - Verify redirect to admin dashboard

3. **Test user login:**
   - Register a test user at http://localhost:8501
   - Go back to http://localhost:8500
   - Login as user
   - Verify redirect to user app

4. **Bookmark the login portal:**
   - http://localhost:8500 is your main entry point!

## ✅ Summary

You now have a **professional, role-based authentication system** with:
- ✅ Single login page for all users
- ✅ Automatic role detection
- ✅ Smart redirection to appropriate dashboard
- ✅ Token-based security
- ✅ Beautiful UI with gradient design
- ✅ Full error handling
- ✅ Easy-to-use batch scripts

**Your users will love this streamlined login experience!** 🚀
