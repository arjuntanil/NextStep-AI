"""
Quick test to check AI Career Advisor response time
"""
import time
import requests

print("🧪 Testing AI Career Advisor Performance\n")
print("=" * 60)

# Test configuration
API_URL = "http://127.0.0.1:8000/career-advice-ai"
test_query = "Tell me about a career in Data Science"

print(f"\n📝 Test Query: {test_query}")
print(f"🔗 API Endpoint: {API_URL}\n")

# Test with different max_length settings
test_configs = [
    {"max_length": 150, "temperature": 0.7, "name": "Short (150 tokens)"},
    {"max_length": 250, "temperature": 0.7, "name": "Medium (250 tokens)"},
    {"max_length": 350, "temperature": 0.7, "name": "Long (350 tokens)"},
]

for config in test_configs:
    print(f"\n{'='*60}")
    print(f"🧪 Test: {config['name']}")
    print(f"   Max Length: {config['max_length']} tokens")
    print(f"   Temperature: {config['temperature']}")
    print("-" * 60)
    
    try:
        payload = {
            "text": test_query,
            "max_length": config['max_length'],
            "temperature": config['temperature']
        }
        
        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=180)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            advice = data.get("advice", "")
            
            print(f"✅ SUCCESS")
            print(f"   ⏱️  Response Time: {elapsed_time:.2f} seconds")
            print(f"   📊 Response Length: {len(advice)} characters")
            print(f"   📊 Model Used: {data.get('model_used', 'Unknown')}")
            print(f"\n   📝 Response Preview:")
            print(f"   {advice[:200]}...")
            
            # Performance rating
            if elapsed_time < 30:
                print(f"   🏆 Performance: EXCELLENT (< 30s)")
            elif elapsed_time < 60:
                print(f"   ⚡ Performance: GOOD (30-60s)")
            elif elapsed_time < 90:
                print(f"   ⚠️  Performance: ACCEPTABLE (60-90s)")
            else:
                print(f"   ❌ Performance: SLOW (> 90s)")
                
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"   Error: {response.json()}")
            
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT after 180 seconds")
    except Exception as e:
        print(f"❌ ERROR: {e}")

print("\n" + "=" * 60)
print("🏁 Performance Test Complete!")
print("\n💡 Recommendations:")
print("   - For fastest responses: Use max_length=150-200")
print("   - For balanced: Use max_length=200-250")
print("   - For detailed: Use max_length=250-350")
print("\n📊 Expected response times:")
print("   - CPU (no GPU): 30-90 seconds")
print("   - GPU: 5-15 seconds")
print("=" * 60)
