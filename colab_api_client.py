"""
Colab API Client - Use Your Google Colab Trained Model Without Downloading
===========================================================================

This script connects your local NextStepAI system to the model trained in Google Colab.
No need to download the 700MB model - just use the Colab API directly!

Setup Steps:
1. Train model in Google Colab (follow GOOGLE_COLAB_TRAINING_COMPLETE_GUIDE.md)
2. Run the BONUS section code to create the public API
3. Copy the ngrok URL (e.g., https://xxxx.ngrok-free.app)
4. Update COLAB_API_URL below
5. Use this client in your backend_api.py

"""

import requests
from typing import Optional
import time

class ColabAPIClient:
    """Client to interact with Google Colab trained model via API"""
    
    def __init__(self, colab_url: str):
        """
        Initialize Colab API client
        
        Args:
            colab_url: The ngrok URL from Colab (e.g., https://xxxx.ngrok-free.app)
        """
        self.colab_url = colab_url.rstrip('/')
        self.health_endpoint = f"{self.colab_url}/health"
        self.advice_endpoint = f"{self.colab_url}/career-advice"
        
    def check_health(self) -> dict:
        """
        Check if Colab API is healthy and running
        
        Returns:
            dict: Health status information
        """
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            response.raise_for_status()
            return {
                "status": "healthy",
                "data": response.json()
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_career_advice(self, question: str, max_retries: int = 3) -> Optional[str]:
        """
        Get career advice from Colab model
        
        Args:
            question: User's career question
            max_retries: Number of retry attempts if request fails
            
        Returns:
            str: Career advice response, or None if failed
        """
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.advice_endpoint,
                    json={"question": question},
                    timeout=30  # Model generation can take 10-20 seconds
                )
                response.raise_for_status()
                
                data = response.json()
                if data.get("status") == "success":
                    return data.get("advice")
                else:
                    print(f"⚠️  API returned error: {data.get('error')}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"⚠️  Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: {e}")
                return None
        
        print("❌ All retry attempts failed")
        return None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # ⚠️ UPDATE THIS WITH YOUR COLAB NGROK URL
    COLAB_API_URL = "https://xxxx-xx-xxx-xxx-xx.ngrok-free.app"
    
    print("="*70)
    print("🚀 Testing Colab API Connection")
    print("="*70)
    
    # Initialize client
    client = ColabAPIClient(COLAB_API_URL)
    
    # Check health
    print("\n1️⃣ Checking API health...")
    health = client.check_health()
    
    if health["status"] == "healthy":
        print("✅ API is healthy!")
        print(f"   Model: {health['data'].get('model')}")
        print(f"   Device: {health['data'].get('device')}")
    else:
        print(f"❌ API is not healthy: {health.get('error')}")
        print("\n📋 Troubleshooting:")
        print("   1. Make sure Colab notebook is running")
        print("   2. Check if you ran the BONUS API creation code")
        print("   3. Verify the ngrok URL is correct")
        exit(1)
    
    # Test career advice
    print("\n2️⃣ Testing career advice generation...")
    test_questions = [
        "I love DevOps",
        "What is software development?",
        "Tell me about cloud engineering"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(test_questions)}: {question}")
        print('─'*70)
        
        advice = client.get_career_advice(question)
        
        if advice:
            print(f"✅ Response received:")
            print(advice[:300] + "..." if len(advice) > 300 else advice)
            
            # Quality check
            has_skills = any(w in advice.lower() for w in ['skill', 'learn', 'technology'])
            has_questions = any(w in advice.lower() for w in ['question', 'interview'])
            print(f"\n   {'✅' if has_skills else '❌'} Contains skills")
            print(f"   {'✅' if has_questions else '⚠️'} Contains interview questions")
        else:
            print("❌ Failed to get response")
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE!")
    print("="*70)
    print("\n📋 Next steps:")
    print("   1. If tests passed, integrate with backend_api.py")
    print("   2. Update ProductionLLMCareerAdvisor to use ColabAPIClient")
    print("   3. Start your FastAPI backend")
    print("\n💡 Your model is now accessible without downloading!")
