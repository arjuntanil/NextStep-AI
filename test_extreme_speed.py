"""
Quick Speed Test for AI Career Advisor - EXTREME MODE
Run this to verify your optimizations are working
"""

import requests
import time

BACKEND_URL = "http://127.0.0.1:8000"

def test_ai_speed():
    """Test AI Career Advisor speed with extreme optimizations"""
    
    print("="*70)
    print("⚡ EXTREME SPEED TEST - AI Career Advisor")
    print("="*70)
    
    test_cases = [
        {
            "name": "FASTEST (60 tokens)",
            "question": "What skills for Data Science?",
            "max_length": 60,
            "target_time": 8
        },
        {
            "name": "FAST (80 tokens)",
            "question": "Tell me about machine learning careers",
            "max_length": 80,
            "target_time": 12
        },
        {
            "name": "BALANCED (100 tokens)",
            "question": "How do I become a software engineer?",
            "max_length": 100,
            "target_time": 18
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/3: {test['name']}")
        print(f"Question: {test['question']}")
        print(f"Max Length: {test['max_length']} tokens")
        print(f"Target Time: < {test['target_time']}s")
        print("="*70)
        
        payload = {
            "text": test['question'],
            "max_length": test['max_length'],
            "temperature": 0.5
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{BACKEND_URL}/career-advice-ai",
                json=payload,
                timeout=45
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                advice = data.get('advice', 'No advice returned')
                
                # Calculate tokens (rough estimate)
                word_count = len(advice.split())
                
                status = "✅ PASS" if elapsed_time < test['target_time'] else "⚠️ SLOW"
                
                print(f"\n{status}")
                print(f"⏱️  Response Time: {elapsed_time:.2f}s (target: <{test['target_time']}s)")
                print(f"📝 Word Count: {word_count} words")
                print(f"\n💬 Response:\n{advice[:200]}...")
                
                results.append({
                    "test": test['name'],
                    "time": elapsed_time,
                    "target": test['target_time'],
                    "passed": elapsed_time < test['target_time'],
                    "words": word_count
                })
                
            else:
                print(f"❌ ERROR: Status {response.status_code}")
                print(f"Response: {response.text}")
                results.append({
                    "test": test['name'],
                    "time": elapsed_time,
                    "target": test['target_time'],
                    "passed": False,
                    "words": 0
                })
                
        except requests.exceptions.Timeout:
            print(f"❌ TIMEOUT after 45 seconds!")
            results.append({
                "test": test['name'],
                "time": 45,
                "target": test['target_time'],
                "passed": False,
                "words": 0
            })
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "test": test['name'],
                "time": 0,
                "target": test['target_time'],
                "passed": False,
                "words": 0
            })
        
        # Wait between tests
        if i < len(test_cases):
            print(f"\n⏳ Waiting 5 seconds before next test...")
            time.sleep(5)
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    for result in results:
        status = "✅" if result['passed'] else "❌"
        print(f"{status} {result['test']}: {result['time']:.2f}s (target: <{result['target']}s)")
    
    print("\n" + "="*70)
    print(f"RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your AI is LIGHTNING FAST! ⚡")
    elif passed >= total - 1:
        print("⚠️ Almost there! Minor tweaking needed.")
    else:
        print("❌ Needs more optimization.")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    print("\n🚀 Starting Speed Test...")
    print("Make sure backend is running on http://127.0.0.1:8000\n")
    
    try:
        results = test_ai_speed()
        print("\n✅ Test completed!")
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
