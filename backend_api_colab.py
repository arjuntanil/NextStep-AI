"""
Backend API with Colab Integration - Use Colab Trained Model Without Download
==============================================================================

This is a modified version of backend_api.py that uses the Google Colab API
instead of loading a local model. NO DOWNLOAD NEEDED!

Setup:
1. Train model in Google Colab
2. Create the public API (BONUS section in guide)
3. Update COLAB_API_URL below with your ngrok URL
4. Run this backend: python -m uvicorn backend_api_colab:app --port 8000

"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from typing import Optional
import os

# ============================================================================
# ⚠️ UPDATE THIS WITH YOUR COLAB NGROK URL
# ============================================================================
COLAB_API_URL = os.getenv("COLAB_API_URL", "https://xxxx-xx-xxx-xxx-xx.ngrok-free.app")

# ============================================================================
# COLAB API CLIENT
# ============================================================================

class ColabAPIClient:
    """Client to interact with Google Colab trained model"""
    
    def __init__(self, colab_url: str):
        self.colab_url = colab_url.rstrip('/')
        self.health_endpoint = f"{self.colab_url}/health"
        self.advice_endpoint = f"{self.colab_url}/career-advice"
    
    def check_health(self) -> bool:
        """Check if Colab API is available"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_career_advice(self, question: str) -> Optional[str]:
        """Get career advice from Colab model"""
        try:
            response = requests.post(
                self.advice_endpoint,
                json={"question": question},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success":
                return data.get("advice")
            return None
            
        except Exception as e:
            print(f"Error calling Colab API: {e}")
            return None


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="NextStepAI Career Advisor (Colab Powered)")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Colab client
colab_client = ColabAPIClient(COLAB_API_URL)

# Request/Response models
class CareerQuestion(BaseModel):
    question: str

class CareerAdviceResponse(BaseModel):
    question: str
    advice: str
    source: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    colab_healthy = colab_client.check_health()
    
    return {
        "status": "running",
        "message": "NextStepAI Career Advisor API (Colab Powered)",
        "colab_status": "connected" if colab_healthy else "disconnected",
        "colab_url": COLAB_API_URL,
        "note": "No model download required - using Colab API!"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    colab_healthy = colab_client.check_health()
    
    if not colab_healthy:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Colab API is not available",
                "troubleshooting": [
                    "Make sure your Colab notebook is running",
                    "Verify the BONUS API creation code was executed",
                    "Check that COLAB_API_URL is correct",
                    "Colab sessions last ~12 hours, you may need to restart"
                ]
            }
        )
    
    return {
        "status": "healthy",
        "colab_connected": True,
        "colab_url": COLAB_API_URL
    }

@app.post("/career-advice-ai", response_model=CareerAdviceResponse)
async def get_career_advice(request: CareerQuestion):
    """
    Get AI-powered career advice using Colab trained model
    
    No local model download required!
    """
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Check Colab API health
    if not colab_client.check_health():
        raise HTTPException(
            status_code=503,
            detail="Colab API is not available. Please ensure your Colab notebook is running."
        )
    
    # Get advice from Colab
    print(f"📡 Calling Colab API for: {question}")
    advice = colab_client.get_career_advice(question)
    
    if not advice:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate career advice. Please try again."
        )
    
    print(f"✅ Response received ({len(advice)} chars)")
    
    return CareerAdviceResponse(
        question=question,
        advice=advice,
        source="Google Colab GPT-2-Medium (Fine-tuned)"
    )


# ============================================================================
# STARTUP MESSAGE
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*70)
    print("🚀 NextStepAI Career Advisor API - Colab Powered")
    print("="*70)
    print(f"Colab API URL: {COLAB_API_URL}")
    
    # Check Colab connection
    print("\n🔍 Checking Colab connection...")
    if colab_client.check_health():
        print("✅ Colab API is connected and healthy!")
        print("✅ No local model required - using Colab directly!")
    else:
        print("⚠️  Colab API is not responding")
        print("\n📋 To fix this:")
        print("   1. Go to your Colab notebook")
        print("   2. Run the BONUS section code to create the API")
        print("   3. Copy the ngrok URL")
        print("   4. Update COLAB_API_URL in this file or set environment variable:")
        print(f"      export COLAB_API_URL=https://your-ngrok-url.ngrok-free.app")
        print("   5. Restart this backend")
    
    print("\n" + "="*70)
    print("📡 API Ready at: http://localhost:8000")
    print("📚 Docs at: http://localhost:8000/docs")
    print("="*70 + "\n")


# ============================================================================
# RUN INSTRUCTIONS
# ============================================================================

"""
To run this backend:

1. Update COLAB_API_URL with your ngrok URL from Colab

2. Run the backend:
   python -m uvicorn backend_api_colab:app --port 8000

3. Test it:
   curl -X POST "http://localhost:8000/career-advice-ai" \
        -H "Content-Type: application/json" \
        -d '{"question": "I love DevOps"}'

4. Or use Python:
   import requests
   response = requests.post(
       "http://localhost:8000/career-advice-ai",
       json={"question": "I love DevOps"}
   )
   print(response.json()["advice"])

Benefits:
✅ No 700MB model download
✅ No local GPU required
✅ Always use the latest trained model
✅ Easy to retrain and update
✅ Works on any machine
"""
