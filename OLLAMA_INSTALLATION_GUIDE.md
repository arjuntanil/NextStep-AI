# Ollama Installation Guide for Windows

## Quick Install Steps

### Step 1: Download Ollama
1. Go to https://ollama.ai/download
2. Click **"Download for Windows"**
3. Download the installer (OllamaSetup.exe)

### Step 2: Install Ollama
1. Run the downloaded **OllamaSetup.exe**
2. Follow the installation wizard
3. Ollama will be installed to: `C:\Users\<YourUsername>\AppData\Local\Programs\Ollama\`
4. **Important:** The installer will add Ollama to your PATH automatically
5. After installation, **restart your PowerShell terminal** or **restart your computer**

### Step 3: Verify Installation
Open a **new PowerShell terminal** and run:
```powershell
ollama --version
```

You should see something like:
```
ollama version is 0.x.x
```

### Step 4: Pull Mistral 7B Q4 Model
```powershell
ollama pull mistral:7b-q4
```

This will download the model (~4GB). It may take several minutes depending on your internet speed.

Expected output:
```
pulling manifest
pulling 8934d96d3f08... 100% ▕████████████████▏ 4.1 GB
pulling 8c17c2ebb0ea... 100% ▕████████████████▏ 7.0 KB
pulling 7c23fb36d801... 100% ▕████████████████▏ 4.8 KB
pulling 2e0493f67d0c... 100% ▕████████████████▏   59 B
pulling fa304d675061... 100% ▕████████████████▏   91 B
pulling 42347cd80dc8... 100% ▕████████████████▏  485 B
verifying sha256 digest
writing manifest
success
```

### Step 5: Verify Model Installation
```powershell
ollama list
```

You should see:
```
NAME              ID              SIZE    MODIFIED
mistral:7b-q4     8934d96d3f08    4.1 GB  X minutes ago
```

### Step 6: Test the Model (Optional)
```powershell
ollama run mistral:7b-q4
```

Type a test prompt:
```
>>> Hello, how are you?
```

Type `/bye` to exit.

## Troubleshooting

### Issue: "ollama is not recognized"

**Cause:** Ollama is either not installed or PATH is not updated.

**Solution 1:** Restart PowerShell
Close all PowerShell terminals and open a new one. The PATH should be updated.

**Solution 2:** Restart Computer
If restarting PowerShell doesn't work, restart your computer.

**Solution 3:** Manual PATH Check
1. Open **Environment Variables**:
   - Press `Win + X` → System → Advanced system settings
   - Click "Environment Variables"
2. Check if `C:\Users\<YourUsername>\AppData\Local\Programs\Ollama` is in PATH
3. If not, add it manually and restart PowerShell

**Solution 4:** Use Full Path
```powershell
& "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" --version
```

### Issue: Model download is slow

**Solution:** 
- Use a faster internet connection
- Download during off-peak hours
- The model is 4.1GB, so it will take time

### Issue: Model download fails

**Solution:**
1. Check your internet connection
2. Try again: `ollama pull mistral:7b-q4`
3. Check disk space (need at least 5GB free)

### Issue: Ollama service not running

**Solution:**
Ollama should start automatically. If not:
```powershell
# Check if Ollama is running
Get-Process ollama -ErrorAction SilentlyContinue

# If not running, start it manually by running any ollama command
ollama list
```

## Alternative: Use Without Ollama

If you don't want to install Ollama right now, **all other features still work perfectly**:

✅ **Resume Analyzer** - Works without Ollama
✅ **AI Career Advisor** - Works without Ollama (uses fine-tuned GPT-2)
✅ **Live Job Postings** - Works without Ollama
✅ **User History** - Works without Ollama

❌ **RAG Coach** - Requires Ollama Mistral 7B Q4

The application will show helpful error messages in the RAG Coach tab if Ollama is not installed.

## After Installation

Once Ollama is installed and the model is pulled:

1. Restart your backend server:
   ```powershell
   E:/NextStepAI/career_coach/Scripts/python.exe -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
   ```

2. Start the frontend:
   ```powershell
   streamlit run app.py
   ```

3. Go to the **"🧑‍💼 RAG Coach"** tab and start using it!

## System Requirements

- **OS:** Windows 10/11 (64-bit)
- **RAM:** 8GB minimum (16GB recommended for Mistral 7B)
- **Disk Space:** 5GB free space for the model
- **Internet:** Required for initial download

## Model Information

- **Model:** Mistral 7B Instruct Q4
- **Size:** 4.1 GB (4-bit quantization)
- **RAM Usage:** ~4-6 GB when running
- **Context Window:** 8K tokens (8192)
- **Speed:** Fast inference on CPU (Q4 quantization)

## Links

- Ollama Official Website: https://ollama.ai
- Ollama Download: https://ollama.ai/download
- Ollama GitHub: https://github.com/ollama/ollama
- Mistral Model Card: https://ollama.ai/library/mistral

---

**Need Help?** Check the troubleshooting section or refer to `RAG_COACH_SETUP_GUIDE.md` for more details.
