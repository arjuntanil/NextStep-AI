"""
Automatic Ollama Setup and RAG Coach Launcher
This script handles everything automatically - no manual PATH setup needed!
"""

import os
import sys
import subprocess
import time

print("🚀 Automatic RAG Coach Setup\n")
print("=" * 70)

# Step 1: Find Ollama
print("\n📍 Step 1: Locating Ollama installation...")
ollama_path = None
possible_paths = [
    os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Ollama', 'ollama.exe'),
    r'C:\Program Files\Ollama\ollama.exe',
    os.path.join('C:\\Users', os.environ.get('USERNAME', ''), 'AppData', 'Local', 'Programs', 'Ollama', 'ollama.exe'),
]

for path in possible_paths:
    if os.path.exists(path):
        ollama_path = path
        print(f"   ✅ Found Ollama at: {path}")
        break

if not ollama_path:
    print("   ❌ Ollama not installed!")
    print("\n   📥 Please install Ollama:")
    print("   1. Download from: https://ollama.ai/download")
    print("   2. Run the installer (OllamaSetup.exe)")
    print("   3. Restart this script")
    sys.exit(1)

# Step 2: Check Ollama version
print("\n🔍 Step 2: Checking Ollama version...")
try:
    result = subprocess.run([ollama_path, '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        version = result.stdout.strip()
        print(f"   ✅ {version}")
    else:
        print("   ⚠️  Could not get version, but Ollama is installed")
except Exception as e:
    print(f"   ⚠️  Error checking version: {e}")

# Step 3: Check for installed models
print("\n📦 Step 3: Checking for Mistral models...")
try:
    result = subprocess.run([ollama_path, 'list'], capture_output=True, text=True, timeout=10)
    
    if result.returncode == 0:
        output = result.stdout
        print("   Installed models:")
        
        mistral_found = False
        for line in output.split('\n')[1:]:  # Skip header
            if line.strip():
                print(f"      • {line.strip()}")
                if 'mistral' in line.lower():
                    mistral_found = True
        
        if mistral_found:
            print("   ✅ Mistral model is available!")
        else:
            print("   ⚠️  No Mistral model found")
            print("\n   💡 Pulling mistral:7b-instruct (recommended)...")
            print("   📥 This will download ~4GB and take 5-15 minutes...")
            
            # Offer to pull model
            response = input("\n   Pull mistral:7b-instruct now? (y/n): ")
            if response.lower() == 'y':
                print("\n   🔄 Downloading model...")
                pull_result = subprocess.run(
                    [ollama_path, 'pull', 'mistral:7b-instruct'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if pull_result.returncode == 0:
                    print("   ✅ Model downloaded successfully!")
                else:
                    print(f"   ❌ Download failed: {pull_result.stderr}")
                    sys.exit(1)
            else:
                print("   ⏭️  Skipping model download")
                print("   ℹ️  RAG Coach will not work without a Mistral model")
                print(f"   💡 To pull later, run: {ollama_path} pull mistral:7b-instruct")
    else:
        print(f"   ⚠️  Could not list models: {result.stderr}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Step 4: Test Ollama connection
print("\n🔌 Step 4: Testing Ollama LLM connection...")
try:
    from langchain_community.llms import Ollama
    
    # Get first available mistral model
    result = subprocess.run([ollama_path, 'list'], capture_output=True, text=True, timeout=5)
    model_name = None
    
    if 'mistral:7b-instruct-q4_k_m' in result.stdout.lower():
        model_name = 'mistral:7b-instruct-q4_K_M'
    elif 'mistral:7b-instruct' in result.stdout.lower():
        model_name = 'mistral:7b-instruct'
    elif 'mistral' in result.stdout.lower():
        model_name = 'mistral'
    
    if model_name:
        print(f"   🤖 Testing with model: {model_name}")
        llm = Ollama(model=model_name, temperature=0.5)
        
        response = llm.invoke("Say 'Hello' in one word.")
        print(f"   ✅ Ollama is responding!")
        print(f"   📝 Test response: {response.strip()[:50]}")
    else:
        print("   ⚠️  No Mistral model available for testing")
        
except Exception as e:
    print(f"   ⚠️  Connection test failed: {e}")
    print("   💡 This is OK - the model might still be downloading")

# Step 5: Setup complete
print("\n" + "=" * 70)
print("✅ Setup Complete!\n")
print("📝 Next steps:")
print("   1. Start backend:")
print(f"      {sys.executable} -m uvicorn backend_api:app --reload\n")
print("   2. Start frontend (in new terminal):")
print(f"      {sys.executable.replace('python.exe', 'activate.bat')}")
print(f"      streamlit run app.py\n")
print("   3. Open browser: http://localhost:8501")
print("   4. Go to '🧑‍💼 RAG Coach' tab")
print("   5. Upload PDFs and start asking questions!")
print("\n" + "=" * 70)
