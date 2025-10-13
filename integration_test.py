#!/usr/bin/env python3
"""
Integration Test for Updated Backend with Fine-tuned Model
Tests the integration of the fine-tuned career advisor into backend_api.py and app.py
"""

import requests
import json
import time
import subprocess
import sys
from pathlib import Path

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing Backend Imports...")
    try:
        import backend_api
        print("✅ backend_api imports successfully")
        
        # Test key components
        from backend_api import FinetunedCareerAdvisor, initialize_finetuned_model
        print("✅ FinetunedCareerAdvisor class available")
        
        # Check if fine-tuned model files exist
        model_path = Path("./career-advisor-finetuned/checkpoint-25")
        if model_path.exists():
            print(f"✅ Fine-tuned model found at {model_path}")
        else:
            print(f"⚠️ Fine-tuned model not found at {model_path}")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_initialization():
    """Test fine-tuned model initialization"""
    print("\n🤖 Testing Model Initialization...")
    try:
        from backend_api import FinetunedCareerAdvisor
        
        advisor = FinetunedCareerAdvisor()
        print(f"✅ FinetunedCareerAdvisor created, device: {advisor.device}")
        print(f"Model loaded: {advisor.is_loaded}")
        
        # Test loading (this might take time)
        print("Loading model (this may take a moment)...")
        advisor.load_model()
        
        if advisor.is_loaded:
            print("✅ Model loaded successfully!")
            
            # Test generation
            test_question = "What skills do I need for data science?"
            print(f"\nTesting generation with: '{test_question}'")
            advice = advisor.generate_advice(test_question, max_length=50, temperature=0.7)
            print(f"✅ Generated advice: {advice[:100]}...")
            
            return True
        else:
            print("❌ Model failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return False

def test_backend_startup():
    """Test backend startup process"""
    print("\n🚀 Testing Backend Startup...")
    try:
        from backend_api import on_startup
        print("Calling on_startup() function...")
        on_startup()
        print("✅ Backend startup completed successfully")
        return True
    except Exception as e:
        print(f"❌ Startup error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints (requires backend to be running)"""
    print("\n🌐 Testing API Endpoints...")
    
    base_url = "http://localhost:8000"
    endpoints_to_test = [
        ("/model-status", "GET"),
        ("/career-advice-ai", "POST"),
        ("/query-career-path/", "POST")
    ]
    
    print("Note: This test requires the backend server to be running on port 8000")
    print("Start the backend with: python -m uvicorn backend_api:app --reload")
    
    for endpoint, method in endpoints_to_test:
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            else:  # POST
                payload = {"text": "What are the key skills for data science?"}
                response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {method} {endpoint} - Status: {response.status_code}")
            else:
                print(f"⚠️ {method} {endpoint} - Status: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {method} {endpoint} - Cannot connect (server not running?)")
        except Exception as e:
            print(f"❌ {method} {endpoint} - Error: {e}")

def main():
    print("🎯 NextStepAI Fine-tuned Integration Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("backend_api.py").exists():
        print("❌ Please run this script from the NextStepAI project directory")
        return False
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Model Initialization", test_model_initialization),
        ("Backend Startup", test_backend_startup),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # API endpoint test (optional)
    print(f"\n{'='*20} API Endpoints (Optional) {'='*20}")
    test_api_endpoints()
    
    # Summary
    print(f"\n{'='*20} Test Summary {'='*20}")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your fine-tuned integration is working correctly!")
        print("\nNext steps:")
        print("1. Start backend: python -m uvicorn backend_api:app --reload")
        print("2. Start frontend: streamlit run app.py")
        print("3. Test the enhanced AI Career Advisor in the web interface")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)