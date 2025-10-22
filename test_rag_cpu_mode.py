"""
Test RAG Coach with CPU-only Ollama configuration
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_rag_status():
    """Check RAG Coach status"""
    print("\n[TEST 1/3] Checking RAG Coach status...")
    response = requests.get(f"{BASE_URL}/rag-coach/status")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_rag_query_direct():
    """Test RAG Coach direct query (no PDFs needed)"""
    print("\n[TEST 2/3] Testing direct LLM query (CPU mode)...")
    
    payload = {
        "question": "What are the top 3 skills for a Python developer?",
        "use_context": False
    }
    
    print(f"Sending query: {payload['question']}")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/rag-coach/query",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n[OK] Answer received:")
            print(f"  {result['answer'][:200]}...")
            print(f"\n  Source Documents: {len(result.get('source_documents', []))}")
            return True
        else:
            print(f"[ERROR] {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False

def test_ollama_direct():
    """Test Ollama directly to verify CPU mode"""
    print("\n[TEST 3/3] Testing Ollama directly...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": "Say 'Hello from CPU mode!' in one sentence.",
                "stream": False,
                "options": {
                    "num_predict": 20,
                    "temperature": 0.5
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Ollama Response: {result.get('response', '')[:100]}")
            return True
        else:
            print(f"[ERROR] Ollama returned status {response.status_code}")
            print(f"Details: {response.text}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Ollama test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Coach CPU Mode Test")
    print("=" * 60)
    
    # Test 1: Status
    status = test_rag_status()
    
    # Test 2: Ollama direct
    ollama_ok = test_ollama_direct()
    
    if not ollama_ok:
        print("\n[ERROR] Ollama is not responding. Make sure it's running in CPU mode.")
        print("Run: ollama serve")
        exit(1)
    
    # Test 3: RAG Query
    query_ok = test_rag_query_direct()
    
    print("\n" + "=" * 60)
    if query_ok:
        print("  [OK] RAG Coach is working in CPU mode!")
    else:
        print("  [ERROR] RAG Coach test failed")
    print("=" * 60)
