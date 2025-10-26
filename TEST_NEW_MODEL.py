"""
Quick test script to verify LLM_FineTuned model loads and generates
Run this BEFORE starting the backend to catch any issues early
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import sys

def test_model():
    print("=" * 60)
    print("🔍 TESTING NEW LLM_FineTuned MODEL")
    print("=" * 60)
    
    model_path = "./LLM_FineTuned"
    
    # Step 1: Load model
    print(f"\n📂 Loading model from: {model_path}")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        print("✅ Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return False
    
    # Step 2: Test generation
    print("\n💬 Testing generation with sample question...")
    question = "Tell me about Data Scientist career path and required skills"
    input_text = f"<|startoftext|>Career Question: {question}\n\nProfessional Career Advice:\n"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
    
    print(f"⏱️  Generating response (target: < 2 seconds)...")
    start_time = time.time()
    
    try:
        outputs = model.generate(
            **inputs,
            min_new_tokens=150,
            max_new_tokens=300,
            temperature=0.75,
            do_sample=True,
            top_p=0.92,
            top_k=40,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            use_cache=True
        )
        
        elapsed = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "Professional Career Advice:" in response:
            response = response.split("Professional Career Advice:")[-1].strip()
        
        word_count = len(response.split())
        
        print(f"\n✅ Generation SUCCESSFUL!")
        print(f"⏱️  Time: {elapsed:.2f} seconds {'✅' if elapsed < 2 else '⚠️ (slower than target)'}")
        print(f"📝 Length: {word_count} words {'✅' if 150 <= word_count <= 300 else '⚠️ (outside target range)'}")
        print(f"\n{'='*60}")
        print("📄 GENERATED RESPONSE:")
        print(f"{'='*60}")
        print(response[:500] + ("..." if len(response) > 500 else ""))
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during generation: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 Model test PASSED! Ready to start backend.")
        print("\nNext steps:")
        print("1. Start backend: python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload")
        print("2. Start frontend: streamlit run app.py")
        sys.exit(0)
    else:
        print("\n❌ Model test FAILED! Check errors above.")
        sys.exit(1)
