# 🎯 How to Test Your Trained Model in Google Colab

## ✅ You Have 2 Options for Testing

---

## 🚀 OPTION 1: Quick & Simple Testing (RECOMMENDED)

### **Use this right after training completes in the SAME Colab session**

#### Step 1: Copy the Quick Test Code
Open file: `COLAB_QUICK_TEST.py` in your local PC

#### Step 2: Paste in New Colab Cell
In your Google Colab notebook (same session where you trained):
1. Create a **NEW CELL** below your training code
2. Paste the entire content of `COLAB_QUICK_TEST.py`
3. Click **Run** ▶️

#### What You'll Get:
- ✅ **3 Automatic Tests**: DevOps, Data Science, Software Development
- ✅ **Interactive Mode**: Ask your own questions
- ✅ **Quality Analysis**: See word count, skills, interview prep
- ✅ **Simple Interface**: Just type and test!

#### Example Usage:
```
🧪 RUNNING QUICK TESTS
📌 TEST 1: DevOps Career
❓ Question: I love to become a DevOps engineer

💡 Career Advice:
[Your model's response here...]

📊 Quality Check:
   Words: 245
   Contains Skills: ✅
   Contains Interview Prep: ✅

💬 INTERACTIVE MODE
❓ Your Question: What skills do I need for AI?
⏳ Generating advice...
[Response here...]
```

---

## 💪 OPTION 2: Comprehensive Testing (For Advanced Users)

### **Use this for detailed performance analysis**

#### Step 1: Copy the Advanced Test Code
Open file: `colab_model_test_user.py` in your local PC

#### Step 2: Paste in New Colab Cell
Same process as Option 1, but you'll get a full menu system

#### What You'll Get:
- ✅ **Pre-defined Tests**: 10 common career questions
- ✅ **Interactive Mode**: Full conversation interface
- ✅ **Batch Testing**: Test multiple custom questions
- ✅ **A/B Temperature Testing**: Compare different generation styles
- ✅ **Performance Benchmark**: Speed and quality metrics

#### Main Menu:
```
🎯 CAREER ADVISOR MODEL - TESTING MENU

Choose a testing mode:
   1. 🧪 Pre-defined Tests (10 common questions)
   2. 💬 Interactive Mode (ask your own questions)
   3. 📋 Batch Test (test custom questions)
   4. 🔬 A/B Temperature Test (compare generation styles)
   5. ⚡ Performance Benchmark (speed & quality)
   6. 🚪 Exit

👉 Enter your choice (1-6):
```

---

## 📋 Complete Testing Workflow

### **AFTER Training Completes in Colab:**

#### ✅ Immediate Testing (Same Session):
```python
# Training just finished, model/tokenizer are loaded

# STEP 1: Paste Quick Test Code in new cell
# STEP 2: Run the cell
# STEP 3: Test with automatic questions
# STEP 4: Use interactive mode to ask your questions
```

#### ✅ Later Testing (New Session):
If you want to test later or saved model to Google Drive:

```python
# Load saved model first
from google.colab import drive
drive.mount('/content/drive')

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load from Drive
model_path = "/content/drive/MyDrive/NextStepAI_Training/career-advisor-trained-model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("✅ Model loaded!")

# NOW paste the testing code in a new cell
```

---

## 🎯 Testing Checklist

Use this to evaluate your model:

### **Basic Quality Checks:**
- [ ] Generates responses without errors
- [ ] Responses are 100-400 words (good length)
- [ ] Contains relevant skills and technologies
- [ ] Includes interview preparation tips
- [ ] Well-structured (bullets, sections)
- [ ] No technical errors (e.g., "Spring Boot + Django")

### **Domain-Specific Checks:**
- [ ] **DevOps**: Mentions Docker, Kubernetes, CI/CD
- [ ] **Data Science**: Mentions Python, ML, pandas, NumPy
- [ ] **Web Dev**: Mentions HTML, CSS, JavaScript, React/Angular
- [ ] **Cloud**: Mentions AWS, Azure, GCP, infrastructure
- [ ] **Security**: Mentions encryption, authentication, testing

### **Accuracy Checks:**
- [ ] No fake technology names (e.g., "Kite Database")
- [ ] No framework confusion (e.g., Spring Boot is Java, not Python)
- [ ] No salary hallucinations (e.g., ₹5 LPA = $0.50)
- [ ] No wrong tool descriptions (e.g., Postman for backend development)

### **Expected Results (15 Epochs, 1e-5 LR):**
- ✅ Validation Loss: < 0.7 (excellent) or < 0.9 (very good)
- ✅ Quality Score: 80%+ on most questions
- ✅ 4-5 out of 5 tests should be "Excellent"
- ✅ No technical errors detected

---

## 💡 Testing Tips

### **1. Temperature Settings:**
```python
# More focused, conservative responses
ask_career_advisor("Your question", temperature=0.5)

# Balanced (recommended)
ask_career_advisor("Your question", temperature=0.7)

# More creative, diverse responses
ask_career_advisor("Your question", temperature=0.9)
```

### **2. Question Phrasing:**
**Good Questions:**
- ✅ "I love to become a DevOps engineer"
- ✅ "What skills do I need for data science?"
- ✅ "How do I prepare for software developer interviews?"

**Better Results With Specific Questions:**
- ⭐ "I love DevOps and have 2 years of experience in Linux. What should I learn next?"
- ⭐ "I'm a beginner in data science. What's the learning path for me?"

### **3. Evaluating Responses:**
**Excellent Response (80%+ quality):**
- Contains 5+ relevant skills
- Lists 3+ interview questions
- Well-structured with sections
- 150-350 words
- No technical errors

**Good Response (60-79% quality):**
- Contains 3-4 skills
- Mentions interview preparation
- Some structure
- 100-200 words
- Minor issues

**Poor Response (<60% quality):**
- Too short (<100 words)
- Missing key information
- No structure
- Contains errors

---

## 🚀 Quick Start Guide

### **Copy This Into Colab (After Training):**

```python
# ============================================================================
# SIMPLEST POSSIBLE TEST - Just 3 lines!
# ============================================================================

def test(q):
    """Ask a question, get career advice!"""
    input_text = f"<|startoftext|>### Question: {q}\n\n### Answer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=350, temperature=0.7, top_p=0.92, top_k=50, 
                             do_sample=True, repetition_penalty=1.3, no_repeat_ngram_size=3, 
                             pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    advice = response.split("### Answer:")[1].strip() if "### Answer:" in response else response
    print(f"\n💡 {advice}\n")
    return advice

# Try it!
test("I love to become a DevOps engineer")
test("What is data science?")
test("How do I become a Full Stack Developer?")
```

---

## 📊 Sample Output

### **What Good Output Looks Like:**

```
❓ Question: I love to become a DevOps engineer

💡 Career Advice:

Great choice! DevOps engineers are in high demand in India's tech industry. 
Here's your roadmap:

### Essential Skills:
• **Docker & Kubernetes**: Container orchestration for deployment
• **CI/CD Pipelines**: Jenkins, GitLab CI, GitHub Actions
• **Cloud Platforms**: AWS, Azure, or Google Cloud
• **Infrastructure as Code**: Terraform, Ansible
• **Linux Administration**: Command line, shell scripting
• **Version Control**: Git, GitHub workflows

### Interview Preparation:
Common questions include:
1. Explain the difference between Docker and Kubernetes
2. How do you handle failed deployments?
3. Describe a CI/CD pipeline you've built
4. What monitoring tools have you used?

### Salary Range in India:
Entry-level: ₹4-6 LPA
Mid-level (2-4 years): ₹8-15 LPA
Senior (5+ years): ₹18-30 LPA

Start with Docker and Git, then move to cloud platforms and automation!

📊 Quality Check:
   Words: 187
   Contains Skills: ✅
   Contains Interview Prep: ✅
```

---

## ✅ Ready to Test!

1. **Finish training** in Google Colab (wait for "✅ TRAINING COMPLETED!")
2. **Create new cell** below training code
3. **Paste testing code** (Option 1 or Option 2)
4. **Run and enjoy!** 🎉

**Files to copy:**
- 📄 `COLAB_QUICK_TEST.py` - Simple, fast testing
- 📄 `colab_model_test_user.py` - Comprehensive testing suite

**Choose based on your needs:**
- Just want to test quickly? → Use `COLAB_QUICK_TEST.py`
- Want detailed analysis? → Use `colab_model_test_user.py`
- Want simplest possible? → Use the 3-line test above

---

## 🎉 Have Fun Testing Your Model!

Your Career Advisor is ready to help with any career questions! 🚀
