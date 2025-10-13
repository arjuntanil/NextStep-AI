"""
Quick API endpoint test for production LLM Career Advisor
"""
import requests
import json
import time

def test_career_advice_endpoint():
    """Test the /career-advice-ai endpoint"""
    
    # Wait for server to fully start
    print("⏳ Waiting for server to start...")
    time.sleep(5)
    
    url = "http://127.0.0.1:8000/career-advice-ai"
    
    test_cases = [
        "I love DevOps",
        "Tell me about cloud engineering",
        "What skills do I need for data science?",
        "I want to become a software developer"
    ]
    
    print("\n" + "="*60)
    print("🧪 TESTING PRODUCTION LLM API ENDPOINT")
    print("="*60 + "\n")
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n{'─'*60}")
        print(f"Test {i}/{len(test_cases)}: {question}")
        print('─'*60)
        
        try:
            response = requests.post(
                url,
                json={"question": question},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: {response.status_code}")
                print(f"\n📝 Response Preview ({len(data.get('advice', ''))} chars):")
                print(data.get('advice', '')[:400] + "...")
                
                # Check for skills and interview questions
                advice = data.get('advice', '')
                has_skills = any(keyword in advice.lower() for keyword in ['skill', 'skills', 'learn', 'knowledge'])
                has_questions = any(keyword in advice.lower() for keyword in ['question', 'interview', 'ask'])
                
                print(f"\n📊 Content Check:")
                print(f"   {'✅' if has_skills else '❌'} Contains skills/learning content")
                print(f"   {'✅' if has_questions else '❌'} Contains interview questions")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   {response.text[:200]}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Server not running")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    print("\n" + "="*60)
    print("✅ API ENDPOINT TESTING COMPLETE")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_career_advice_endpoint()
    exit(0 if success else 1)
