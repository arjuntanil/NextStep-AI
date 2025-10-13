# ============================================================================
# 🎯 QUICK USER TESTING - Career Advisor Model (Google Colab)
# ============================================================================
# Paste this code in a NEW CELL after training completes!
# ============================================================================

import torch

print("\n" + "="*70)
print("🎯 CAREER ADVISOR - USER TESTING MODE")
print("="*70)

# Check if model is loaded (from training session)
try:
    device
    model
    tokenizer
    print("✅ Model is ready!")
    print(f"✅ Device: {device}")
except:
    print("❌ Model not found! Make sure you've trained the model first.")
    raise Exception("Run training code first!")

# ============================================================================
# SIMPLE TESTING FUNCTION
# ============================================================================

def ask_career_advisor(question, show_details=False):
    """Ask the Career Advisor any question!"""
    
    # Prepare input
    input_text = f"<|startoftext|>### Question: {question}\n\n### Answer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate advice
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.7,
            top_p=0.92,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Get response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    advice = response.split("### Answer:")[1].strip() if "### Answer:" in response else response
    
    # Display
    print("\n" + "="*70)
    print(f"❓ Question: {question}")
    print("="*70)
    print(f"\n💡 Career Advice:\n")
    print(advice)
    print("\n" + "="*70)
    
    # Optional quality analysis
    if show_details:
        words = len(advice.split())
        has_skills = any(w in advice.lower() for w in ['skill', 'learn', 'technology'])
        has_interview = any(w in advice.lower() for w in ['question', 'interview', 'prepare'])
        
        print("\n📊 Quality Check:")
        print(f"   Words: {words}")
        print(f"   Contains Skills: {'✅' if has_skills else '❌'}")
        print(f"   Contains Interview Prep: {'✅' if has_interview else '⚠️'}")
        print("="*70)
    
    return advice

# ============================================================================
# 🎯 QUICK TEST - Try These Examples!
# ============================================================================

print("\n" + "="*70)
print("🧪 RUNNING QUICK TESTS")
print("="*70)

# Test 1: DevOps
print("\n📌 TEST 1: DevOps Career")
ask_career_advisor("I love to become a DevOps engineer", show_details=True)

# Test 2: Data Science
print("\n📌 TEST 2: Data Science Career")
ask_career_advisor("I want to become a Data Scientist", show_details=True)

# Test 3: General
print("\n📌 TEST 3: Software Development")
ask_career_advisor("What is software development?", show_details=True)

print("\n" + "="*70)
print("✅ QUICK TESTS COMPLETE!")
print("="*70)

# ============================================================================
# 💬 INTERACTIVE MODE - Ask Your Own Questions!
# ============================================================================

print("\n" + "="*70)
print("💬 INTERACTIVE MODE - ASK YOUR OWN QUESTIONS!")
print("="*70)
print("\n📝 Instructions:")
print("   • Ask any career-related question")
print("   • Type 'stop' to exit")
print("   • Type 'examples' for sample questions")
print("="*70)

# Sample questions
examples = [
    "What skills do I need for cloud computing?",
    "How do I prepare for technical interviews?",
    "Tell me about Full Stack Development",
    "What is the best programming language to learn?",
    "I'm interested in cybersecurity, where should I start?"
]

while True:
    print("\n" + "─"*70)
    user_question = input("\n❓ Your Question (or 'stop' to exit): ").strip()
    
    if user_question.lower() == 'stop':
        print("\n👋 Thanks for testing! Goodbye!")
        break
    
    if user_question.lower() == 'examples':
        print("\n📚 Example Questions:")
        for i, ex in enumerate(examples, 1):
            print(f"   {i}. {ex}")
        continue
    
    if not user_question:
        print("⚠️  Please enter a question!")
        continue
    
    # Generate and display
    print("\n⏳ Generating advice...")
    ask_career_advisor(user_question, show_details=False)
    
    # Quick feedback
    feedback = input("\n💭 Helpful? (y/n): ").strip().lower()
    if feedback == 'y':
        print("✅ Great! Ask another question or type 'stop' to exit.")
    else:
        print("⚠️  Try rephrasing for better results!")

print("\n" + "="*70)
print("🎉 TESTING SESSION COMPLETE!")
print("="*70)
