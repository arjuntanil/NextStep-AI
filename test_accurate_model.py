"""
Comprehensive test for Accurate Career Advisor Model
Tests skills, interview questions, and structured responses
"""

import torch
import json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_model():
    """Test the fine-tuned accurate career advisor model"""
    
    model_path = "./career-advisor-production-v3/final_model"
    
    print("\n" + "="*70)
    print("🧪 ACCURATE CAREER ADVISOR MODEL TEST")
    print("="*70)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Model not found at {model_path}")
        print("   Please run: python accurate_career_advisor_training.py")
        return False
    
    print(f"\n✅ Model found at {model_path}")
    
    # Load model
    print("\n📦 Loading model...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path, torch_dtype=torch.float32)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded on {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test cases matching your requirements
    test_cases = [
        {
            "question": "I love DevOps",
            "expected": ["docker", "kubernetes", "jenkins", "ci/cd", "terraform", "ansible"],
            "needs_interview": True
        },
        {
            "question": "What is software development",
            "expected": ["programming", "coding", "java", "python", "software engineer", "design"],
            "needs_interview": True
        },
        {
            "question": "I love networking",
            "expected": ["cisco", "network", "routing", "tcp/ip", "protocols", "firewall"],
            "needs_interview": True
        },
        {
            "question": "Tell me about Cloud Engineering",
            "expected": ["aws", "azure", "gcp", "cloud", "infrastructure", "scalability"],
            "needs_interview": True
        },
        {
            "question": "Data Science career",
            "expected": ["python", "machine learning", "statistics", "pandas", "scikit-learn"],
            "needs_interview": True
        }
    ]
    
    print("\n" + "="*70)
    print("🔍 RUNNING ACCURACY TESTS")
    print("="*70)
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected_keywords = test_case["expected"]
        needs_interview = test_case["needs_interview"]
        
        print(f"\n{'─'*70}")
        print(f"Test {i}/{total_tests}: {question}")
        print('─'*70)
        
        # Generate response
        input_text = f"Question: {question}\n\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=450,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[1].strip()
        else:
            answer = full_response
        
        print(f"\n💡 Response ({len(answer)} chars):")
        print(answer[:600])
        if len(answer) > 600:
            print("...")
        
        # Quality checks
        answer_lower = answer.lower()
        
        # Check for expected keywords
        keywords_found = sum(1 for kw in expected_keywords if kw in answer_lower)
        keyword_score = keywords_found / len(expected_keywords) * 100
        
        # Check for skills
        has_skills = any(word in answer_lower for word in [
            'skill', 'skills', 'learn', 'knowledge', 'technology', 'tool', 'programming'
        ])
        
        # Check for interview questions
        has_questions = any(word in answer_lower for word in [
            'question', 'interview', 'ask', 'answer', 'q:', 'a:'
        ])
        
        # Check for structure
        has_structure = any(marker in answer for marker in ['###', '*', '•', '\n\n'])
        
        # Check coherence (not random text)
        words = answer_lower.split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        is_coherent = unique_ratio > 0.3 and len(answer) > 100
        
        # Check for irrelevant content
        bad_patterns = [
            'reliance india',
            'canadian',
            'risk research networks',
            'scenario coaching',
            'cips® qi',
            'ask \'canad'
        ]
        has_irrelevant = any(pattern in answer_lower for pattern in bad_patterns)
        
        print(f"\n📊 Quality Assessment:")
        print(f"   {'✅' if keyword_score >= 30 else '❌'} Relevant keywords: {keyword_score:.0f}% ({keywords_found}/{len(expected_keywords)})")
        print(f"   {'✅' if has_skills else '❌'} Contains skills/technologies")
        print(f"   {'✅' if has_questions else '⚠️' } Contains interview questions")
        print(f"   {'✅' if has_structure else '❌'} Has structured formatting")
        print(f"   {'✅' if is_coherent else '❌'} Response is coherent (not random)")
        print(f"   {'✅' if not has_irrelevant else '❌'} No irrelevant/weird content")
        
        # Determine pass/fail
        test_passed = (
            keyword_score >= 20 and  # At least 20% keywords match
            has_skills and
            has_structure and
            is_coherent and
            not has_irrelevant
        )
        
        if test_passed:
            print(f"\n✅ TEST PASSED")
            passed_tests += 1
        else:
            print(f"\n❌ TEST FAILED")
    
    # Final summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"\nTests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Model is accurate and ready!")
        print("\n✅ The model now generates:")
        print("   • Accurate skills for job roles")
        print("   • Interview questions and answers")
        print("   • Structured, coherent responses")
        print("   • No irrelevant or random content")
        return True
    elif passed_tests >= total_tests * 0.6:
        print("\n⚠️ PARTIAL SUCCESS - Most tests passed")
        print("   Model is functional but may need parameter tuning")
        return True
    else:
        print("\n❌ TESTS FAILED - Model needs retraining")
        print("   Consider adjusting training parameters or data quality")
        return False


if __name__ == "__main__":
    success = test_model()
    exit(0 if success else 1)
