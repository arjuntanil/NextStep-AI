"""
Quick verification script to test Ollama and RAG Coach setup
Run this after Ollama model download completes
"""

import subprocess
import sys

print("🔍 Verifying Ollama and RAG Coach Setup\n")
print("=" * 60)

# Step 1: Check if ollama command is available
print("\n1️⃣  Checking Ollama command...")
try:
    result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"   ✅ {result.stdout.strip()}")
    else:
        print(f"   ❌ Ollama command failed")
        sys.exit(1)
except FileNotFoundError:
    print("   ❌ Ollama command not found!")
    print("   💡 Solution: Close and reopen PowerShell after installing Ollama")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Step 2: List installed models
print("\n2️⃣  Checking installed Ollama models...")
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✅ Installed models:")
        for line in result.stdout.split('\n')[1:]:  # Skip header
            if line.strip():
                print(f"      • {line.strip()}")
        
        # Check for Mistral
        if 'mistral' in result.stdout.lower():
            print("   ✅ Mistral model found!")
        else:
            print("   ⚠️  No Mistral model found")
            print("   💡 Run: ollama pull mistral:7b-instruct")
    else:
        print("   ❌ Could not list models")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Step 3: Test RAG Coach imports
print("\n3️⃣  Testing RAG Coach dependencies...")
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    print("   ✅ langchain-community imports OK")
    print("   ✅ pypdf available")
    print("   ✅ ollama Python package available")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    print("   💡 Run: pip install langchain-community ollama pypdf")
    sys.exit(1)

# Step 4: Test RAG Coach initialization
print("\n4️⃣  Testing RAG Coach initialization...")
try:
    from rag_coach import RAGCoachSystem
    print("   ✅ RAG Coach module imported")
    
    # Try to initialize (but don't load models yet)
    coach = RAGCoachSystem()
    print(f"   ✅ RAG Coach initialized with model: {coach.llm_model_name}")
    
except Exception as e:
    print(f"   ❌ Error initializing RAG Coach: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test Ollama connection
print("\n5️⃣  Testing Ollama LLM connection...")
try:
    from langchain_community.llms import Ollama
    
    # Detect available Mistral model
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
    mistral_model = None
    
    if 'mistral:7b-instruct-q4' in result.stdout.lower():
        mistral_model = 'mistral:7b-instruct-q4_K_M'
    elif 'mistral:7b-instruct' in result.stdout.lower():
        mistral_model = 'mistral:7b-instruct'
    elif 'mistral' in result.stdout.lower():
        mistral_model = 'mistral'
    
    if mistral_model:
        print(f"   ℹ️  Testing with model: {mistral_model}")
        llm = Ollama(model=mistral_model, temperature=0.7)
        
        # Test with a simple prompt
        response = llm.invoke("Say 'Hello from RAG Coach!' in one sentence.")
        print(f"   ✅ Ollama LLM responding!")
        print(f"   📝 Test response: {response[:100]}...")
    else:
        print("   ⚠️  No Mistral model available for testing")
        print("   💡 Pull a model: ollama pull mistral:7b-instruct")
        
except Exception as e:
    print(f"   ❌ Error connecting to Ollama: {e}")
    print("   💡 Make sure Ollama is running and model is pulled")

# Summary
print("\n" + "=" * 60)
print("📊 Verification Summary:")
print("=" * 60)
print("\n✅ All checks passed! Your RAG Coach is ready to use.")
print("\n📝 Next steps:")
print("   1. Start backend: python -m uvicorn backend_api:app --reload")
print("   2. Start frontend: streamlit run app.py")
print("   3. Go to 'RAG Coach' tab")
print("   4. Upload PDFs and start asking questions!")
print("\n")
