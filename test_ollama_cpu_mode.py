"""
Test Ollama CPU Mode - Verify it works without GPU errors
"""

import os
import sys

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OLLAMA_NUM_GPU'] = '0'

print("=" * 60)
print("   TESTING OLLAMA IN CPU MODE")
print("=" * 60)
print()

try:
    from langchain_community.llms import Ollama
    
    print("[1/3] Connecting to Ollama...")
    llm = Ollama(
        model='mistral:7b-instruct',
        temperature=0.5,
        num_ctx=1024,
        num_gpu=0,  # Force CPU
        num_thread=4  # Use 4 CPU threads
    )
    print("[OK] Connected to Ollama")
    print()
    
    print("[2/3] Testing inference (this may take 10-15 seconds)...")
    response = llm.invoke("Hello! What is AI?")
    print("[OK] Inference successful!")
    print()
    
    print("[3/3] Response:")
    print("-" * 60)
    print(response[:200] + "..." if len(response) > 200 else response)
    print("-" * 60)
    print()
    
    print("=" * 60)
    print("   SUCCESS! OLLAMA WORKS IN CPU MODE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run: FIX_RAG_GPU_ERROR.bat")
    print("2. Go to: http://localhost:8501")
    print("3. Test RAG Coach with PDF uploads")
    print()
    
except Exception as e:
    print()
    print("=" * 60)
    print("   ERROR!")
    print("=" * 60)
    print()
    print(f"Error: {e}")
    print()
    print("Solutions:")
    print("1. Make sure Ollama is running:")
    print("   Start -> Search 'Ollama' -> Right-click -> Run")
    print()
    print("2. Or run this command:")
    print("   RESTART_OLLAMA_CPU_MODE.bat")
    print()
    print("3. Verify model is installed:")
    print("   ollama list")
    print()
    print("4. Pull model if missing:")
    print("   ollama pull mistral:7b-instruct")
    print()
    sys.exit(1)
