"""
Manual Model Loader - Load LLM_FineTuned without timeout issues
Run this ONCE to pre-load the model, then start the backend
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🔧 MANUAL MODEL LOADER - LLM_FineTuned (DistilGPT-2)")
print("=" * 70)
print()
print("📌 This script loads the model WITHOUT timeout issues")
print("📌 Run this BEFORE starting the backend for faster startup")
print()

# Check if model exists
from pathlib import Path
model_path = "./LLM_FineTuned"

if not Path(model_path).exists():
    print(f"❌ ERROR: Model folder not found at {model_path}")
    print()
    print("Please ensure LLM_FineTuned folder exists in the root directory")
    input("Press Enter to exit...")
    sys.exit(1)

print(f"✅ Model folder found: {model_path}")
print()

# Import the Production LLM Career Advisor class
print("📦 Importing dependencies...")
try:
    from backend_api import ProductionLLMCareerAdvisor
    print("✅ Dependencies imported")
except Exception as e:
    print(f"❌ ERROR importing dependencies: {e}")
    print()
    print("Make sure you're in the correct virtual environment:")
    print("  .\\career_coach\\Scripts\\Activate.ps1")
    input("Press Enter to exit...")
    sys.exit(1)

print()
print("=" * 70)
print("🚀 LOADING MODEL (This will take 30-60 seconds)")
print("=" * 70)
print()

# Create instance and load model
try:
    advisor = ProductionLLMCareerAdvisor()
    print(f"📂 Model path: {advisor.model_path}")
    print()
    
    # Load the model
    advisor.load_model()
    
    print()
    print("=" * 70)
    print("✅ MODEL LOADED SUCCESSFULLY!")
    print("=" * 70)
    print()
    
    # Test generation
    print("🧪 Testing generation...")
    test_question = "What skills do I need for a Data Scientist role?"
    
    import time
    start = time.time()
    response = advisor.generate_advice(test_question, max_length=200, temperature=0.75)
    elapsed = time.time() - start
    
    word_count = len(response.split())
    
    print()
    print("=" * 70)
    print("📝 TEST RESULTS:")
    print("=" * 70)
    print(f"⏱️  Generation time: {elapsed:.2f} seconds")
    print(f"📏 Response length: {word_count} words")
    print()
    print("Response preview (first 300 chars):")
    print("-" * 70)
    print(response[:300] + ("..." if len(response) > 300 else ""))
    print("-" * 70)
    print()
    
    if elapsed < 5 and word_count > 50:
        print("✅ Model is working well!")
        print()
        print("=" * 70)
        print("🎉 SUCCESS! Model is ready to use")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Keep this model loaded (don't close this window)")
        print("2. Start backend: python -m uvicorn backend_api:app --reload")
        print("3. The model will load MUCH FASTER now")
        print()
        print("NOTE: The model stays in memory until you close this window")
    else:
        print("⚠️  Model loaded but response quality may need tuning")
        print()
    
except Exception as e:
    print()
    print("=" * 70)
    print("❌ ERROR LOADING MODEL")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    
    if "paging file" in str(e).lower() or "memory" in str(e).lower():
        print("💡 MEMORY ISSUE DETECTED")
        print()
        print("Your system doesn't have enough memory to load the model.")
        print()
        print("SOLUTIONS:")
        print("1. Close other applications to free up memory")
        print("2. Increase Windows virtual memory (paging file):")
        print("   - Windows Settings → System → About → Advanced system settings")
        print("   - Performance Settings → Advanced → Virtual memory")
        print("   - Change → Set custom size: 8000-16000 MB")
        print()
        print("3. Use RAG mode instead (works without loading the model)")
        print("   - The backend will automatically use RAG if model fails to load")
        print()
    
    import traceback
    print("Full error details:")
    print(traceback.format_exc())

print()
input("Press Enter to exit...")
