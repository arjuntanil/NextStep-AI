#!/usr/bin/env python3
"""
Fine-tuned Career Advisor Client Example
Demonstrates how to interact with the enhanced backend API
"""

import requests
import json
import asyncio
from typing import List, Dict

class CareerAdvisorClient:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.headers = {"Content-Type": "application/json"}
    
    def get_career_advice(self, question: str, max_length: int = 200, temperature: float = 0.7) -> Dict:
        """Get career advice from fine-tuned model"""
        payload = {
            "question": question,
            "max_length": max_length, 
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/career-advice-ai",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
    
    def compare_models(self, question: str) -> Dict:
        """Compare fine-tuned vs RAG model responses"""
        payload = {"question": question}
        
        try:
            response = requests.post(
                f"{self.api_base_url}/career-advice-compare",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
    
    def get_model_status(self) -> Dict:
        """Check if models are loaded and ready"""
        try:
            response = requests.get(f"{self.api_base_url}/model-status", timeout=10)
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}

def interactive_demo():
    """Interactive demonstration of the career advisor"""
    print("🤖 Fine-tuned Career Advisor Interactive Demo")
    print("=" * 50)
    
    client = CareerAdvisorClient()
    
    # Check model status first
    print("🔍 Checking model status...")
    status = client.get_model_status()
    
    if not status["success"]:
        print(f"❌ Cannot connect to API: {status['error']}")
        print("Make sure the enhanced backend is running on localhost:8000")
        return
    
    model_info = status["data"]
    finetuned_loaded = model_info.get("finetuned_career_advisor", {}).get("loaded", False)
    
    if not finetuned_loaded:
        print("❌ Fine-tuned model not loaded. Please start the enhanced backend first.")
        return
    
    print("✅ Fine-tuned model is ready!")
    print(f"Device: {model_info['finetuned_career_advisor']['device']}")
    print(f"Model: {model_info['finetuned_career_advisor']['base_model']}")
    
    print("\n🎯 Ask your career questions (type 'quit' to exit):")
    
    while True:
        question = input("\n💬 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        print("\n🤔 Thinking...")
        
        # Get advice from fine-tuned model
        result = client.get_career_advice(question, max_length=150, temperature=0.7)
        
        if result["success"]:
            data = result["data"]
            print(f"\n🧠 **{data['model_used']}** (Confidence: {data['confidence']})")
            print(f"📝 **Advice:** {data['advice']}")
        else:
            print(f"❌ Error: {result['error']}")
        
        # Ask if user wants comparison
        compare = input("\n🔄 Compare with RAG model? (y/n): ").strip().lower()
        
        if compare == 'y':
            print("🔍 Getting comparison...")
            comparison = client.compare_models(question)
            
            if comparison["success"] and comparison["data"]["comparison_available"]:
                responses = comparison["data"]["responses"]
                
                print("\n📊 **Model Comparison:**")
                for model_type, response in responses.items():
                    print(f"\n--- {model_type.upper()} ---")
                    print(f"Model: {response['model']}")
                    print(f"Response: {response['advice'][:200]}...")
            else:
                print("❌ Comparison not available")

def batch_test_demo():
    """Test multiple questions in batch"""
    print("🧪 Batch Testing Demo")
    print("=" * 30)
    
    client = CareerAdvisorClient()
    
    test_questions = [
        "What programming languages should I learn for AI/ML?",
        "How do I transition from finance to data science?",
        "What certifications are valuable for cloud computing?",
        "How should I prepare for a technical product manager role?",
        "What are the key skills for cybersecurity professionals?"
    ]
    
    print(f"Testing {len(test_questions)} questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] {question}")
        
        result = client.get_career_advice(question, max_length=100, temperature=0.6)
        
        if result["success"]:
            advice = result["data"]["advice"]
            print(f"✅ {advice[:150]}...")
        else:
            print(f"❌ Error: {result['error']}")

def main():
    print("🚀 NextStepAI Fine-tuned Career Advisor Client")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Interactive Demo")
        print("2. Batch Test Demo") 
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            interactive_demo()
        elif choice == "2":
            batch_test_demo()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()