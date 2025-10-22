# 🚀 FIX FOR PORT 8000 ERROR - RUN THIS NOW!

## ❌ Your Error:
```
ERROR: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions
```

## ✅ SOLUTION (30 seconds):

### Step 1: Close your current terminal

### Step 2: Open NEW terminal (cmd or PowerShell)

### Step 3: Run this ONE command:
```cmd
cd E:\NextStepAI && RESTART_BACKEND.bat
```

**That's it!** The script automatically:
- Kills any process using port 8000
- Starts backend fresh
- Loads all RAG Coach fixes

---

## 🎯 After Backend Starts:

### Open ANOTHER new terminal and run:
```cmd
cd E:\NextStepAI && START_FRONTEND.bat
```

### Then open browser:
```
http://localhost:8501
```

---

## 📝 What Each Script Does:

### RESTART_BACKEND.bat:
- Finds process using port 8000
- Kills it automatically
- Starts backend with --reload
- **Use this when you see port errors!**

### START_BACKEND.bat:
- Just starts backend
- **Use this when port is free**

### START_FRONTEND.bat:
- Starts Streamlit UI
- Opens on port 8501

---

## ⚡ Quick Commands:

### Kill port 8000 manually:
```cmd
for /f "tokens=5" %a in ('netstat -ano ^| findstr ":8000.*LISTENING"') do taskkill /F /PID %a
```

### Then start backend:
```cmd
uvicorn backend_api:app --reload --host 127.0.0.1 --port 8000
```

---

## 🎉 SIMPLE VERSION:

**Just run these 2 files in 2 different terminals:**

**Terminal 1:**
```
RESTART_BACKEND.bat
```

**Terminal 2:**
```
START_FRONTEND.bat
```

**Done!** ✅

Your app is at: http://localhost:8501

---

**All issues are fixed. The RESTART_BACKEND.bat script solves your port error!**
