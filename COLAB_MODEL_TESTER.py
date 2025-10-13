# ============================================================================
# 🎯 COLAB MODEL TESTER - Check Your Trained Career Advisor LLM
# ============================================================================
# PASTE THIS IN A NEW CELL IN COLAB AFTER TRAINING COMPLETES
# ============================================================================

import torch

print("\n" + "="*70)
print("🎯 CAREER ADVISOR MODEL - QUALITY CHECKER")
print("="*70)

# Verify model is loaded
try:
    device
    model
    tokenizer
    print("✅ Model is ready to test!")
    print(f"✅ Device: {device}")
    print(f"✅ Model: GPT2-Medium (fine-tuned)")
except:
    print("❌ ERROR: Model not found!")
    print("   Please run the training code first in a previous cell.")
    raise Exception("Model not loaded. Run training first!")

print("\n" + "="*70)
print("🧪 STARTING QUALITY TESTS")
print("="*70)

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_model(question, show_details=False):
    """
    Test the model with a question and return quality metrics
    """
    # Generate response
    input_text = f"<|startoftext|>### Question: {question}\n\n### Answer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
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
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    advice = response.split("### Answer:")[1].strip() if "### Answer:" in response else response
    
    # Quality checks
    advice_lower = advice.lower()
    word_count = len(advice.split())
    words = advice_lower.split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    
    # Check 1: Skills/Technologies
    has_skills = any(w in advice_lower for w in ['skill', 'skills', 'learn', 'technology', 'technologies', 'knowledge'])
    
    # Check 2: Interview Preparation
    has_interview = any(w in advice_lower for w in ['question', 'interview', 'ask', 'prepare'])
    
    # Check 3: Well-structured
    has_structure = any(marker in advice for marker in ['###', '*', '•', '1.', '2.', '\n-'])
    
    # Check 4: No technical errors
    error_patterns = [
        ('spring boot', 'django'),
        ('₹3–5 lpa', '$0.40'),
        ('kite database', ''),
        ('oracle 9-core', ''),
        ('postman for backend', '')
    ]
    
    has_errors = False
    detected_errors = []
    for pattern1, pattern2 in error_patterns:
        if pattern1 in advice_lower:
            if pattern2 and pattern2 in advice_lower:
                has_errors = True
                detected_errors.append(f"{pattern1}+{pattern2}")
            elif not pattern2:
                has_errors = True
                detected_errors.append(pattern1)
    
    # Check 5: Good length
    is_good_length = 50 < word_count < 400
    
    # Check 6: Coherent
    is_coherent = unique_ratio > 0.4
    
    # Calculate score
    checks = [has_skills, has_interview, has_structure, not has_errors, is_good_length, is_coherent]
    score = sum(checks)
    
    # Grade
    if score >= 5:
        grade = "EXCELLENT ✅"
    elif score >= 4:
        grade = "GOOD ✅"
    elif score >= 3:
        grade = "FAIR ⚠️"
    else:
        grade = "POOR ❌"
    
    # Display results
    print(f"\n{'='*70}")
    print(f"❓ Question: {question}")
    print(f"{'─'*70}")
    print(f"\n💡 Generated Advice:\n")
    print(advice[:300] + "..." if len(advice) > 300 else advice)
    
    if show_details:
        print(f"\n{'─'*70}")
        print("📊 QUALITY ANALYSIS:")
        print(f"{'─'*70}")
        print(f"   {'✅' if has_skills else '❌'} Contains Skills/Technologies")
        print(f"   {'✅' if has_interview else '❌'} Contains Interview Prep")
        print(f"   {'✅' if has_structure else '❌'} Well-Structured Format")
        print(f"   {'✅' if not has_errors else '❌'} No Technical Errors" + (f" (Found: {', '.join(detected_errors)})" if has_errors else ""))
        print(f"   {'✅' if is_good_length else '❌'} Good Length ({word_count} words)")
        print(f"   {'✅' if is_coherent else '❌'} Coherent ({unique_ratio:.1%} unique)")
    
    print(f"\n{'─'*70}")
    print(f"🎯 SCORE: {score}/6 - {grade}")
    print(f"{'='*70}")
    
    return {
        'advice': advice,
        'score': score,
        'grade': grade,
        'word_count': word_count,
        'has_errors': has_errors,
        'errors': detected_errors
    }

# ============================================================================
# AUTOMATIC TESTS - 5 Key Questions
# ============================================================================

print("\n\n" + "="*70)
print("🧪 AUTOMATIC QUALITY TESTS (5 Questions)")
print("="*70)

test_questions = [
    "I love to become a DevOps engineer",
    "I love to become a Data Scientist",
    "What is software development?",
    "Tell me about cloud engineering",
    "I want to learn networking"
]

results = []
for i, question in enumerate(test_questions, 1):
    print(f"\n{'#'*70}")
    print(f"TEST {i}/5")
    print(f"{'#'*70}")
    result = test_model(question, show_details=True)
    results.append(result)

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n\n" + "="*70)
print("📊 OVERALL TEST SUMMARY")
print("="*70)

total_score = sum(r['score'] for r in results)
max_score = len(results) * 6
percentage = (total_score / max_score) * 100

excellent_count = sum(1 for r in results if r['score'] >= 5)
good_count = sum(1 for r in results if r['score'] == 4)
fair_count = sum(1 for r in results if r['score'] == 3)
poor_count = sum(1 for r in results if r['score'] < 3)

errors_found = sum(1 for r in results if r['has_errors'])

print(f"\n📈 Overall Score: {total_score}/{max_score} ({percentage:.1f}%)")
print(f"\n📊 Test Results:")
print(f"   ✅ EXCELLENT (5-6/6): {excellent_count} tests")
print(f"   ✅ GOOD (4/6): {good_count} tests")
print(f"   ⚠️  FAIR (3/6): {fair_count} tests")
print(f"   ❌ POOR (<3/6): {poor_count} tests")
print(f"\n⚠️  Technical Errors Found: {errors_found} tests")

if errors_found > 0:
    print("\n   Errors detected in:")
    for i, r in enumerate(results, 1):
        if r['has_errors']:
            print(f"   - Test {i}: {', '.join(r['errors'])}")

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("\n" + "="*70)
print("🎯 FINAL VERDICT")
print("="*70)

if percentage >= 85 and errors_found == 0:
    print("\n🌟 EXCELLENT - Model is PRODUCTION-READY!")
    print("   ✅ High accuracy across all tests")
    print("   ✅ No technical errors detected")
    print("   ✅ Safe to deploy immediately")
    print("\n📋 Next Steps:")
    print("   1. ✅ Save to Google Drive (Option B)")
    print("   2. ✅ Or download to local PC (Option C)")
    print("   3. ✅ Deploy to production")
    
elif percentage >= 70 and errors_found <= 1:
    print("\n✅ GOOD - Model is usable with minor monitoring")
    print("   ✅ Good accuracy on most tests")
    print("   ⚠️  Minor issues detected (acceptable)")
    print("   ✅ Can deploy but monitor responses")
    print("\n📋 Next Steps:")
    print("   1. ✅ Save to Google Drive or download")
    print("   2. ⚠️  Deploy with monitoring")
    print("   3. 💡 Consider retraining with 18 epochs if issues persist")
    
elif percentage >= 50:
    print("\n⚠️  FAIR - Model needs improvement")
    print("   ⚠️  Moderate accuracy issues")
    print("   ⚠️  Some technical errors detected")
    print("   ❌ NOT recommended for production")
    print("\n📋 Recommended Actions:")
    print("   1. ❌ DO NOT save this model")
    print("   2. 🔄 RETRAIN with these settings:")
    print("      • epochs = 20")
    print("      • learning_rate = 1e-5")
    print("      • batch_size = 2")
    print("   3. 🧪 Run this test again after retraining")
    
else:
    print("\n❌ POOR - Model needs significant retraining")
    print("   ❌ Low accuracy across tests")
    print("   ❌ Multiple technical errors")
    print("   ❌ DO NOT deploy to production")
    print("\n📋 Critical Actions Required:")
    print("   1. ❌ DO NOT save/download this model")
    print("   2. 🔄 RETRAIN with improved settings:")
    print("      • epochs = 20-25")
    print("      • learning_rate = 5e-6 (even lower)")
    print("      • Check training data quality")
    print("   3. 🧪 Run comprehensive validation")

print("\n" + "="*70)

# ============================================================================
# INTERACTIVE MODE - Ask Your Own Questions!
# ============================================================================

print("\n\n" + "="*70)
print("💬 INTERACTIVE MODE - Test Your Own Questions!")
print("="*70)
print("\n📝 Instructions:")
print("   • Type any career question to test")
print("   • Type 'exit' or 'quit' to stop")
print("   • Type 'examples' for sample questions")
print("="*70)

examples = [
    "What skills do I need for Full Stack Development?",
    "How do I prepare for cybersecurity interviews?",
    "Tell me about AI and Machine Learning careers",
    "What is blockchain development?",
    "I'm interested in mobile app development"
]

while True:
    print("\n" + "─"*70)
    user_input = input("\n❓ Your Question (or 'exit' to stop): ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'stop']:
        print("\n👋 Testing complete! Good luck with your Career Advisor!")
        break
    
    if user_input.lower() == 'examples':
        print("\n📚 Example Questions:")
        for i, ex in enumerate(examples, 1):
            print(f"   {i}. {ex}")
        continue
    
    if not user_input:
        print("⚠️  Please enter a question!")
        continue
    
    print("\n⏳ Generating response...")
    result = test_model(user_input, show_details=True)
    
    feedback = input("\n💭 Was this response good? (y/n): ").strip().lower()
    if feedback == 'n':
        print("💡 Tip: Try rephrasing or being more specific!")

print("\n" + "="*70)
print("🎉 MODEL TESTING COMPLETE!")
print("="*70)
print("\n✅ Your Career Advisor LLM has been thoroughly tested!")
print("✅ Review the summary above to decide next steps")
print("✅ Save the model if quality is good (≥70% score)")
print("\n📚 For detailed deployment guide, see HOW_TO_TEST_MODEL.md")
