# ============================================================================
# CAREER ADVISOR MODEL - USER TESTING INTERFACE (For Google Colab)
# Test your trained model interactively!
# ============================================================================

# NOTE: Run this AFTER training is complete in the same Colab session
# If loading from a saved model, see the "Load Saved Model" section below

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("\n" + "="*70)
print("🎯 CAREER ADVISOR MODEL - USER TESTING")
print("="*70)

# ============================================================================
# OPTION 1: Use Already Trained Model (Same Session)
# ============================================================================
# If you just finished training, the model and tokenizer are already loaded!
# Skip to "Testing Functions" section below

# ============================================================================
# OPTION 2: Load Saved Model (New Session or From Drive)
# ============================================================================
# Uncomment this section if you're loading a previously saved model

"""
print("\n📂 Loading saved model...")

# Choose one:
# A) Load from local Colab directory (if in same session)
model_path = "./career-advisor-final"

# B) Load from Google Drive (if saved to Drive)
# from google.colab import drive
# drive.mount('/content/drive')
# model_path = "/content/drive/MyDrive/NextStepAI_Training/career-advisor-trained-model"

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"✅ Model loaded successfully!")
print(f"✅ Device: {device}")
"""

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def generate_career_advice(question: str, temperature: float = 0.7, max_length: int = 400) -> dict:
    """
    Generate career advice for a user question
    
    Args:
        question: User's career question
        temperature: Controls randomness (0.1-1.0). Lower = more focused
        max_length: Maximum words in response
        
    Returns:
        dict with advice, metadata, and quality scores
    """
    # Prepare input
    input_text = f"<|startoftext|>### Question: {question}\n\n### Answer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "### Answer:" in full_response:
        advice = full_response.split("### Answer:")[1].strip()
    else:
        advice = full_response
    
    # Quality checks
    advice_lower = advice.lower()
    word_count = len(advice.split())
    words = advice_lower.split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    
    has_skills = any(w in advice_lower for w in ['skill', 'skills', 'learn', 'technology', 'technologies', 'knowledge', 'practice'])
    has_interview = any(w in advice_lower for w in ['question', 'interview', 'ask', 'prepare', 'preparation'])
    has_structure = any(marker in advice for marker in ['###', '*', '•', '1.', '2.', '\n-'])
    is_good_length = 50 < word_count < 500
    is_coherent = unique_ratio > 0.4
    
    # Calculate quality score
    quality_checks = [has_skills, has_interview, has_structure, is_good_length, is_coherent]
    quality_score = sum(quality_checks) / len(quality_checks) * 100
    
    return {
        "advice": advice,
        "word_count": word_count,
        "unique_ratio": unique_ratio,
        "has_skills": has_skills,
        "has_interview_prep": has_interview,
        "has_structure": has_structure,
        "quality_score": quality_score,
        "quality_grade": "Excellent" if quality_score >= 80 else "Good" if quality_score >= 60 else "Fair"
    }


def display_advice(result: dict, show_analysis: bool = True):
    """Display career advice with optional quality analysis"""
    print("\n" + "="*70)
    print("💡 CAREER ADVICE:")
    print("="*70)
    print(result['advice'])
    
    if show_analysis:
        print("\n" + "─"*70)
        print("📊 QUALITY ANALYSIS:")
        print("─"*70)
        print(f"   Word Count: {result['word_count']}")
        print(f"   Unique Words: {result['unique_ratio']:.1%}")
        print(f"   Contains Skills: {'✅' if result['has_skills'] else '❌'}")
        print(f"   Contains Interview Prep: {'✅' if result['has_interview_prep'] else '❌'}")
        print(f"   Well-Structured: {'✅' if result['has_structure'] else '❌'}")
        print(f"   Quality Score: {result['quality_score']:.0f}% ({result['quality_grade']})")
    print("="*70)


# ============================================================================
# TEST MODE 1: Pre-defined Career Questions
# ============================================================================

def run_predefined_tests():
    """Test model with common career questions"""
    print("\n" + "="*70)
    print("🧪 RUNNING PREDEFINED CAREER QUESTIONS")
    print("="*70)
    
    test_questions = [
        "I love to become a DevOps engineer",
        "I want to become a Data Scientist",
        "What is software development?",
        "Tell me about cloud engineering",
        "I want to learn networking",
        "How do I become a Full Stack Developer?",
        "What skills do I need for cybersecurity?",
        "I'm interested in AI and Machine Learning",
        "Tell me about blockchain development",
        "What is the best career path in IT?"
    ]
    
    print(f"\n🎯 Testing {len(test_questions)} questions...\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'='*70}")
        print(f"❓ Question: {question}")
        
        # Generate advice
        result = generate_career_advice(question, temperature=0.7)
        
        # Display (without quality analysis to keep it clean)
        print(f"\n💡 Answer:")
        print(f"{result['advice'][:300]}..." if len(result['advice']) > 300 else result['advice'])
        print(f"\n📊 Quality: {result['quality_score']:.0f}% ({result['quality_grade']}) | Words: {result['word_count']}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)


# ============================================================================
# TEST MODE 2: Interactive User Mode
# ============================================================================

def interactive_mode():
    """Interactive mode - ask questions one by one"""
    print("\n" + "="*70)
    print("💬 INTERACTIVE CAREER ADVISOR MODE")
    print("="*70)
    print("\n📝 Instructions:")
    print("   • Ask any career-related question")
    print("   • Type 'quit' or 'exit' to stop")
    print("   • Type 'help' for example questions")
    print("="*70)
    
    example_questions = [
        "I love to become a DevOps engineer",
        "What skills do I need for data science?",
        "How do I prepare for software developer interviews?",
        "Tell me about cloud computing careers",
        "What is the salary range for Full Stack Developers in India?"
    ]
    
    while True:
        print("\n" + "─"*70)
        question = input("\n❓ Your Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using Career Advisor! Goodbye!")
            break
        
        if question.lower() == 'help':
            print("\n📚 Example Questions:")
            for i, ex in enumerate(example_questions, 1):
                print(f"   {i}. {ex}")
            continue
        
        if not question:
            print("⚠️  Please enter a question!")
            continue
        
        # Generate and display advice
        print("\n⏳ Generating advice...")
        result = generate_career_advice(question, temperature=0.7)
        display_advice(result, show_analysis=True)
        
        # Ask for feedback
        feedback = input("\n💭 Was this helpful? (yes/no): ").strip().lower()
        if feedback in ['yes', 'y']:
            print("✅ Great! Feel free to ask another question!")
        elif feedback in ['no', 'n']:
            print("⚠️  Sorry about that. Try rephrasing your question for better results.")


# ============================================================================
# TEST MODE 3: Batch Testing with Custom Questions
# ============================================================================

def batch_test(questions: list, temperature: float = 0.7, show_full_advice: bool = False):
    """
    Test multiple custom questions
    
    Args:
        questions: List of questions to test
        temperature: Generation temperature
        show_full_advice: Show full advice or just preview
    """
    print("\n" + "="*70)
    print(f"📋 BATCH TESTING - {len(questions)} Questions")
    print("="*70)
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"Question {i}/{len(questions)}: {question}")
        
        result = generate_career_advice(question, temperature=temperature)
        results.append(result)
        
        if show_full_advice:
            print(f"\n💡 Full Advice:")
            print(result['advice'])
        else:
            print(f"\n💡 Preview:")
            print(result['advice'][:200] + "..." if len(result['advice']) > 200 else result['advice'])
        
        print(f"\n📊 Quality: {result['quality_score']:.0f}% | Words: {result['word_count']} | Grade: {result['quality_grade']}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 BATCH TEST SUMMARY")
    print("="*70)
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    avg_words = sum(r['word_count'] for r in results) / len(results)
    excellent_count = sum(1 for r in results if r['quality_score'] >= 80)
    
    print(f"   Average Quality Score: {avg_quality:.1f}%")
    print(f"   Average Word Count: {avg_words:.0f}")
    print(f"   Excellent Responses: {excellent_count}/{len(results)} ({excellent_count/len(results)*100:.0f}%)")
    print("="*70)
    
    return results


# ============================================================================
# TEST MODE 4: A/B Testing (Different Temperatures)
# ============================================================================

def ab_test_temperature(question: str):
    """Test same question with different temperature settings"""
    print("\n" + "="*70)
    print("🔬 A/B TEMPERATURE TESTING")
    print("="*70)
    print(f"\n❓ Question: {question}\n")
    
    temperatures = [0.3, 0.5, 0.7, 0.9]
    
    for temp in temperatures:
        print(f"\n{'─'*70}")
        print(f"🌡️  Temperature: {temp} ({'More Focused' if temp < 0.6 else 'More Creative'})")
        print(f"{'─'*70}")
        
        result = generate_career_advice(question, temperature=temp)
        print(f"\n💡 Response:")
        print(result['advice'][:250] + "..." if len(result['advice']) > 250 else result['advice'])
        print(f"\n📊 Quality: {result['quality_score']:.0f}% | Words: {result['word_count']}")
    
    print("\n" + "="*70)
    print("💡 Recommendation: Temperature 0.7 usually gives the best balance!")
    print("="*70)


# ============================================================================
# TEST MODE 5: Performance Benchmark
# ============================================================================

def performance_benchmark(num_questions: int = 10):
    """Benchmark model performance and speed"""
    import time
    
    print("\n" + "="*70)
    print(f"⚡ PERFORMANCE BENCHMARK - {num_questions} Questions")
    print("="*70)
    
    test_questions = [
        "I love to become a DevOps engineer",
        "I want to become a Data Scientist",
        "What is software development?",
        "Tell me about cloud engineering",
        "I want to learn networking",
        "How do I become a Full Stack Developer?",
        "What skills do I need for cybersecurity?",
        "I'm interested in AI and Machine Learning",
        "Tell me about blockchain development",
        "What is the best career path in IT?"
    ]
    
    questions = (test_questions * ((num_questions // len(test_questions)) + 1))[:num_questions]
    
    print(f"\n⏳ Running {num_questions} generations...\n")
    
    start_time = time.time()
    quality_scores = []
    
    for i, question in enumerate(questions, 1):
        result = generate_career_advice(question, temperature=0.7)
        quality_scores.append(result['quality_score'])
        
        if i % 5 == 0:
            print(f"   Progress: {i}/{num_questions} completed...")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    avg_quality = sum(quality_scores) / len(quality_scores)
    min_quality = min(quality_scores)
    max_quality = max(quality_scores)
    avg_time_per_question = total_time / num_questions
    
    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS")
    print("="*70)
    print(f"   Total Questions: {num_questions}")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Avg Time/Question: {avg_time_per_question:.2f} seconds")
    print(f"   Avg Quality Score: {avg_quality:.1f}%")
    print(f"   Quality Range: {min_quality:.0f}% - {max_quality:.0f}%")
    print(f"   Device: {device}")
    print("="*70)


# ============================================================================
# MAIN MENU - Choose Testing Mode
# ============================================================================

def main_menu():
    """Main menu to choose testing mode"""
    print("\n" + "="*70)
    print("🎯 CAREER ADVISOR MODEL - TESTING MENU")
    print("="*70)
    print("\nChoose a testing mode:")
    print("   1. 🧪 Pre-defined Tests (10 common questions)")
    print("   2. 💬 Interactive Mode (ask your own questions)")
    print("   3. 📋 Batch Test (test custom questions)")
    print("   4. 🔬 A/B Temperature Test (compare generation styles)")
    print("   5. ⚡ Performance Benchmark (speed & quality)")
    print("   6. 🚪 Exit")
    print("="*70)
    
    while True:
        choice = input("\n👉 Enter your choice (1-6): ").strip()
        
        if choice == '1':
            run_predefined_tests()
            input("\n⏎ Press Enter to return to menu...")
            main_menu()
            break
        
        elif choice == '2':
            interactive_mode()
            input("\n⏎ Press Enter to return to menu...")
            main_menu()
            break
        
        elif choice == '3':
            print("\n📝 Enter your questions (one per line, empty line to finish):")
            custom_questions = []
            while True:
                q = input(f"   Question {len(custom_questions)+1}: ").strip()
                if not q:
                    break
                custom_questions.append(q)
            
            if custom_questions:
                show_full = input("\n💡 Show full advice? (yes/no): ").lower() in ['yes', 'y']
                batch_test(custom_questions, show_full_advice=show_full)
            else:
                print("⚠️  No questions provided!")
            
            input("\n⏎ Press Enter to return to menu...")
            main_menu()
            break
        
        elif choice == '4':
            question = input("\n❓ Enter a question for A/B testing: ").strip()
            if question:
                ab_test_temperature(question)
            else:
                print("⚠️  No question provided!")
            
            input("\n⏎ Press Enter to return to menu...")
            main_menu()
            break
        
        elif choice == '5':
            num = input("\n🔢 How many questions to benchmark? (default: 10): ").strip()
            num_questions = int(num) if num.isdigit() else 10
            performance_benchmark(num_questions)
            
            input("\n⏎ Press Enter to return to menu...")
            main_menu()
            break
        
        elif choice == '6':
            print("\n👋 Thank you for testing! Goodbye!")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1-6.")


# ============================================================================
# AUTO-RUN: Start Main Menu
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Starting Career Advisor Model Testing...")
    print(f"✅ Device: {device}")
    print(f"✅ Model: GPT2-Medium (Fine-tuned)")
    
    main_menu()
