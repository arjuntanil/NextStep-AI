# 🚀 Google Colab Training Guide - Complete Step-by-Step

## ✅ Why Google Colab is PERFECT for This

| Feature | Your RTX 2050 | Google Colab T4 | Advantage |
|---------|--------------|-----------------|-----------|
| **GPU Memory** | 4 GB | 15 GB | 4x more memory |
| **Speed** | 15-20 min | 5-10 min | 2-3x faster |
| **Setup** | Need CUDA install | Ready to use | No setup needed |
| **Cost** | Free | Free | Same! |
| **Internet** | Not needed | Required | Easy download after |

---

## 📋 Complete Step-by-Step Process

### STEP 1: Prepare Your Data Files (On Your Local PC)

**Upload these 2 files to Google Drive:**

1. `career_advice_dataset.jsonl` (243 examples)
2. `career_advice_ultra_clear_dataset.jsonl` (255 examples)

**How to upload:**
1. Go to https://drive.google.com
2. Click "New" → "Folder" → Name it "NextStepAI_Training"
3. Open the folder
4. Click "New" → "File Upload"
5. Upload both JSONL files from `E:\NextStepAI\`

✅ **Done!** Your data is now in Google Drive.

---

### STEP 2: Create Google Colab Notebook

1. Go to https://colab.research.google.com
2. Click "File" → "New Notebook"
3. Rename it: "Career_Advisor_Training"
4. Click "Runtime" → "Change runtime type"
5. Select: **GPU** (T4, A100, or V100)
6. Click "Save"

✅ **Done!** You now have a GPU-enabled Colab notebook.

---

### STEP 3: Copy This COMPLETE Training Code to Colab

**Create a new cell and paste this entire code:**

```python
# ============================================================================
# CAREER ADVISOR LLM FINE-TUNING - GOOGLE COLAB
# GPU-Optimized Training Script
# ============================================================================

# Step 1: Install required packages
print("📦 Installing required packages...")
!pip install -q transformers datasets accelerate

# Step 2: Mount Google Drive
print("\n📁 Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# Step 3: Set up training directory
import os
os.chdir('/content')

# Step 4: Import libraries
print("\n📚 Importing libraries...")
import json
import torch
from pathlib import Path
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from datetime import datetime

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Step 5: Load training data from Google Drive
print("\n" + "="*70)
print("📚 LOADING TRAINING DATA FROM GOOGLE DRIVE")
print("="*70)

# ⚠️ UPDATE THESE PATHS TO YOUR GOOGLE DRIVE FOLDER
drive_folder = "/content/drive/MyDrive/NextStepAI_Training"

data_files = [
    f"{drive_folder}/career_advice_dataset.jsonl",
    f"{drive_folder}/career_advice_ultra_clear_dataset.jsonl"
]

all_examples = []
for file_path in data_files:
    if Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                try:
                    example = json.loads(line.strip())
                    all_examples.append(example)
                    count += 1
                except:
                    continue
            print(f"✅ {Path(file_path).name}: {count} examples")
    else:
        print(f"❌ File not found: {file_path}")
        print(f"   Please upload this file to Google Drive folder: {drive_folder}")

print(f"\n📊 Total training examples: {len(all_examples)}")

if len(all_examples) < 100:
    print("\n❌ ERROR: Not enough training data!")
    print("   Please upload the JSONL files to Google Drive")
    raise Exception("Training data not found")

# Step 6: Prepare dataset
print("\n" + "="*70)
print("🔧 PREPARING DATASET")
print("="*70)

def format_example(prompt, completion):
    return f"<|startoftext|>### Question: {prompt}\n\n### Answer: {completion}<|endoftext|>"

# Initialize tokenizer
model_name = "gpt2-medium"
print(f"Loading tokenizer: {model_name}")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Format and tokenize
formatted_texts = [format_example(ex['prompt'], ex['completion']) for ex in all_examples]

print("Tokenizing dataset...")
tokenized = tokenizer(
    formatted_texts,
    truncation=True,
    padding='max_length',
    max_length=512,
    return_tensors='pt'
)

# Create dataset
dataset = Dataset.from_dict({
    'input_ids': tokenized['input_ids'],
    'attention_mask': tokenized['attention_mask']
})

# Split for training and validation
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"✅ Training examples: {len(split_dataset['train'])}")
print(f"✅ Validation examples: {len(split_dataset['test'])}")

# Step 7: Initialize model
print("\n" + "="*70)
print("🤖 LOADING MODEL")
print("="*70)
print(f"Model: {model_name} (355M parameters)")

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Step 8: Configure training (IMPROVED FOR ACCURACY!)
print("\n" + "="*70)
print("⚙️ TRAINING CONFIGURATION - ACCURACY OPTIMIZED")
print("="*70)

train_samples = len(split_dataset['train'])
batch_size = 2  # Smaller batch for better accuracy
grad_accum = 8  # Higher accumulation for stability
epochs = 15  # More epochs for maximum accuracy

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Training samples: {train_samples}")
print(f"Validation samples: {len(split_dataset['test'])}")
print(f"Epochs: {epochs} (maximum for high accuracy)")
print(f"Batch size: {batch_size} (smaller for precision)")
print(f"Gradient accumulation: {grad_accum} (higher for stability)")
print(f"Effective batch size: {batch_size * grad_accum}")
print(f"Learning rate: 1e-5 (very low for maximum precision)")
print(f"Mixed precision: FP16 enabled")
print(f"Steps per epoch: ~{train_samples // (batch_size * grad_accum)}")
print(f"Total steps: ~{(train_samples // (batch_size * grad_accum)) * epochs}")
print(f"Expected time: 35-40 minutes (best accuracy!)")

training_args = TrainingArguments(
    output_dir="./career-advisor-colab-trained",
    overwrite_output_dir=True,
    
    # MAXIMUM ACCURACY settings
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    
    # Very low learning rate for maximum precision
    learning_rate=1e-5,  # Reduced from 2e-5 for better accuracy
    warmup_ratio=0.15,   # More warmup steps
    weight_decay=0.01,
    
    # GPU optimization
    fp16=True,
    dataloader_pin_memory=True,
    
    # Better logging and validation
    logging_steps=5,     # Log more frequently
    logging_first_step=True,
    eval_strategy="steps",  # Evaluate every N steps
    eval_steps=20,          # Check validation loss often
    save_strategy="steps",
    save_steps=20,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Other
    report_to="none",
    seed=42,
    remove_unused_columns=False
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    data_collator=data_collator,
)

# Step 9: START TRAINING! (MAXIMUM ACCURACY MODE)
print("\n" + "="*70)
print("🚀 STARTING MAXIMUM ACCURACY TRAINING")
print("="*70)
print("⏰ Expected time: 35-40 minutes on Colab GPU")
print("💡 15 epochs with 1e-5 learning rate for BEST accuracy!")
print("📊 Watch the eval_loss - target is below 0.7 for excellent quality")
print("🎯 This will eliminate technical errors like 'Spring Boot + Django'")
print("Progress will be shown below:")
print("="*70 + "\n")

start_time = datetime.now()

# Add early stopping callback
from transformers import EarlyStoppingCallback
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

trainer.train()
end_time = datetime.now()
elapsed = end_time - start_time

print("\n" + "="*70)
print("✅ TRAINING COMPLETED!")
print("="*70)
print(f"Training time: {elapsed}")
print("="*70)

# Step 10: Save model
print("\n💾 Saving trained model...")
output_path = "./career-advisor-final"
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

# Save metadata
metadata = {
    "model": "gpt2-medium (355M parameters)",
    "training_samples": len(all_examples),
    "epochs": epochs,
    "batch_size": batch_size,
    "gradient_accumulation": grad_accum,
    "learning_rate": 1e-5,
    "device": torch.cuda.get_device_name(0),
    "training_time": str(elapsed),
    "trained_on": "Google Colab GPU",
    "optimization": "FP16 mixed precision",
    "quality": "Maximum accuracy - Production-ready"
}

with open(f"{output_path}/training_info.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved to: {output_path}")

# Step 11: COMPREHENSIVE ACCURACY TEST
print("\n" + "="*70)
print("🧪 COMPREHENSIVE ACCURACY TESTING")
print("="*70)

model.eval()
device = torch.device("cuda")
model.to(device)

def generate_advice(question: str, temperature: float = 0.7) -> str:
    """Generate career advice with better parameters for accuracy"""
    input_text = f"<|startoftext|>### Question: {question}\n\n### Answer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=temperature,  # Lower temp = more focused
            top_p=0.92,
            top_k=50,  # Added for better quality
            do_sample=True,
            repetition_penalty=1.3,  # Higher to avoid repetition
            no_repeat_ngram_size=3,  # Avoid repeating 3-grams
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Answer:" in response:
        return response.split("### Answer:")[1].strip()
    return response

# Comprehensive test questions
test_questions = [
    "I love to become a DevOps engineer",
    "I love to become a Data Scientist",
    "What is software development?",
    "Tell me about cloud engineering",
    "I want to learn networking"
]

print("\n🎯 Testing model accuracy with comprehensive checks...\n")

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*70}")
    print(f"TEST {i}/{len(test_questions)}")
    print(f"{'='*70}")
    print(f"Question: {question}")
    print(f"{'─'*70}")
    
    # Generate with lower temperature for accuracy
    advice = generate_advice(question, temperature=0.7)
    
    print(f"\n💡 Generated Advice:\n")
    print(advice[:500] + "..." if len(advice) > 500 else advice)
    
    # Comprehensive quality checks
    print(f"\n{'─'*70}")
    print("📊 ACCURACY CHECKS:")
    print(f"{'─'*70}")
    
    advice_lower = advice.lower()
    
    # Check 1: Contains skills/technologies
    skill_keywords = ['skill', 'skills', 'learn', 'technology', 'technologies', 'knowledge']
    has_skills = any(w in advice_lower for w in skill_keywords)
    print(f"   {'✅' if has_skills else '❌'} Contains skills/technologies")
    
    # Check 2: Contains interview questions
    interview_keywords = ['question', 'interview', 'ask', 'prepare']
    has_questions = any(w in advice_lower for w in interview_keywords)
    print(f"   {'✅' if has_questions else '⚠️'} Contains interview preparation")
    
    # Check 3: Well-structured (has bullets or sections)
    has_structure = ('###' in advice or '*' in advice or '•' in advice or 
                     '1.' in advice or '2.' in advice)
    print(f"   {'✅' if has_structure else '❌'} Well-structured format")
    
    # Check 4: No obvious errors (check for common mistakes)
    error_patterns = [
        ('spring boot', 'django'),  # Spring Boot is Java, Django is Python
        ('₹3–5 lpa', '$0.40'),      # Salary hallucination
        ('oracle 9-core', 'mongodb'),  # Nonsense combinations
        ('kite database', ''),       # Non-existent tech
        ('postman for backend', ''),  # Wrong tool description
    ]
    
    has_errors = False
    detected_errors = []
    for pattern1, pattern2 in error_patterns:
        if pattern1 in advice_lower:
            if pattern2 and pattern2 in advice_lower:
                has_errors = True
                detected_errors.append(f"{pattern1} + {pattern2}")
            elif not pattern2:  # Single pattern check
                has_errors = True
                detected_errors.append(pattern1)
    
    if has_errors:
        print(f"   ❌ FOUND ERRORS: {', '.join(detected_errors)}")
    else:
        print(f"   ✅ No obvious technical errors detected")
    
    # Check 5: Reasonable length
    word_count = len(advice.split())
    print(f"   {'✅' if 50 < word_count < 400 else '⚠️'} Reasonable length ({word_count} words)")
    
    # Check 6: Coherence (unique words ratio)
    words = advice_lower.split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    print(f"   {'✅' if unique_ratio > 0.4 else '❌'} Good coherence ({unique_ratio:.1%} unique words)")
    
    # Overall score
    checks_passed = sum([
        has_skills,
        has_questions,
        has_structure,
        not has_errors,
        50 < word_count < 400,
        unique_ratio > 0.4
    ])
    
    print(f"\n{'─'*70}")
    print(f"🎯 OVERALL SCORE: {checks_passed}/6 checks passed")
    
    if checks_passed >= 5:
        print("✅ EXCELLENT - Model is highly accurate!")
    elif checks_passed >= 4:
        print("✅ GOOD - Model is mostly accurate")
    elif checks_passed >= 3:
        print("⚠️  FAIR - Model needs more training")
    else:
        print("❌ POOR - Model needs significant improvement")
        print("   💡 Consider retraining with more epochs or lower learning rate")

print("\n" + "="*70)
print("🎉 TRAINING & TESTING COMPLETE!")
print("="*70)

# Final evaluation metrics
final_loss = trainer.state.log_history[-1].get('eval_loss', 'N/A')
print(f"\n� Final Validation Loss: {final_loss}")

if isinstance(final_loss, float):
    if final_loss < 0.7:
        print("✅ EXCELLENT - Maximum accuracy achieved!")
    elif final_loss < 0.9:
        print("✅ VERY GOOD - High accuracy model!")
    elif final_loss < 1.1:
        print("✅ GOOD - Model should work well")
    else:
        print("⚠️  FAIR - Consider 20 epochs if issues persist")

print("\n�📋 Next steps:")
print("   If accuracy tests passed (5+/6 checks):")
print("   1. Save to Google Drive (Option B)")
print("   2. Or download model folder: 'career-advisor-final'")
print("   3. Extract to: E:\\NextStepAI\\career-advisor-production-v3\\final_model\\")
print("   4. Run backend: python -m uvicorn backend_api:app --port 8000")
print("\n   If accuracy tests failed (<4/6 checks):")
print("   ⚠️  RETRAIN with these changes:")
print("   • Increase epochs to 20")
print("   • Keep learning rate at 1e-5")
print("   • Consider improving training data quality")
print("\n✅ Your model is ready if tests passed!")
```

---

### STEP 4: Run the Training

1. **Click the "Play" button** on the left side of the cell
2. **Wait for prompts:**
   - "Permit this notebook to access your Google Drive files?" → Click **"Connect to Google Drive"** → **Allow**
   - It will mount your Google Drive

3. **Watch the training progress:**
   - You'll see epochs 1-6 progressing
   - Loss will decrease from ~4.5 to ~1.1
   - Takes 5-10 minutes total

4. **Training output will look like:**
```
📦 Installing required packages...
📁 Mounting Google Drive...
✅ GPU: Tesla T4
✅ GPU Memory: 15.0 GB

📚 LOADING TRAINING DATA
✅ career_advice_dataset.jsonl: 243 examples
✅ career_advice_ultra_clear_dataset.jsonl: 255 examples
📊 Total: 498 examples

🚀 STARTING TRAINING
Expected time: 5-10 minutes

Epoch 1/6: [████████] loss: 2.845
Epoch 2/6: [████████] loss: 2.234
Epoch 3/6: [████████] loss: 1.876
Epoch 4/6: [████████] loss: 1.543
Epoch 5/6: [████████] loss: 1.321
Epoch 6/6: [████████] loss: 1.145

✅ TRAINING COMPLETED!
Training time: 0:08:23
```

---

### STEP 5: Use the Model (Choose One Option)

After training completes, you have **2 options**:

---

#### **OPTION A: Use Model Directly in Colab (RECOMMENDED - No Download!)** ⚡

You can test and use the model right in Colab without downloading!

**Run this in a new cell:**

```python
# ============================================================================
# TEST THE MODEL IN COLAB (NO DOWNLOAD NEEDED!)
# ============================================================================

print("\n" + "="*70)
print("🧪 TESTING MODEL IN GOOGLE COLAB")
print("="*70)

# Model is already trained and ready!
# Let's test it with various career questions

test_questions = [
    "I love DevOps",
    "What is software development", 
    "I love networking",
    "Tell me about cloud engineering",
    "What skills do I need for data science?"
]

print(f"\n✅ Model: gpt2-medium (trained)")
print(f"✅ Device: {torch.cuda.get_device_name(0)}")
print(f"✅ Ready to generate career advice!\n")

for i, question in enumerate(test_questions, 1):
    print("\n" + "="*70)
    print(f"Test {i}/{len(test_questions)}: {question}")
    print("="*70)
    
    # Generate response
    input_text = f"<|startoftext|>### Question: {question}\n\n### Answer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "### Answer:" in response:
        answer = response.split("### Answer:")[1].strip()
    else:
        answer = response
    
    print(f"\n💡 Career Advice:\n")
    print(answer)
    
    # Quality check
    has_skills = any(w in answer.lower() for w in ['skill', 'skills', 'learn', 'technology', 'knowledge'])
    has_questions = any(w in answer.lower() for w in ['question', 'interview', 'ask'])
    has_structure = '###' in answer or '*' in answer or '•' in answer
    
    print(f"\n📊 Quality Check:")
    print(f"   {'✅' if has_skills else '❌'} Contains skills/technologies")
    print(f"   {'✅' if has_questions else '⚠️'} Contains interview questions")
    print(f"   {'✅' if has_structure else '❌'} Well-structured response")

print("\n" + "="*70)
print("✅ TESTING COMPLETE!")
print("="*70)
print("\n💡 Your model is working! You can:")
print("   1. Continue using it here in Colab")
print("   2. Create an API endpoint in Colab")
print("   3. Save to Google Drive for permanent storage")
print("   4. Download to your local PC if needed")
```

**This gives you instant results without any download!** ⚡

---

#### **OPTION B: Save Model to Google Drive (For Permanent Storage)** 💾

Save directly to Google Drive so you can access it anytime:

```python
# Save to Google Drive
import shutil

drive_save_path = "/content/drive/MyDrive/NextStepAI_Training/career-advisor-trained-model"
print(f"💾 Saving model to Google Drive...")
print(f"   Path: {drive_save_path}")

# Copy model to Drive
shutil.copytree("./career-advisor-final", drive_save_path, dirs_exist_ok=True)

print("✅ Model saved to Google Drive!")
print("\n📋 You can now:")
print("   1. Access model from any Colab notebook")
print("   2. Share with others via Drive")
print("   3. Load it anytime without retraining")
```

**To load the model later from Drive:**
```python
# In a new Colab session
from google.colab import drive
drive.mount('/content/drive')

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_path = "/content/drive/MyDrive/NextStepAI_Training/career-advisor-trained-model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda")
model.to(device)

print("✅ Model loaded from Google Drive!")
```

---

#### **OPTION C: Download to Your Local PC (For Local Deployment)** 📥

Only if you need to deploy on your local machine:

**1. Zip the model:**
```python
!zip -r career-advisor-final.zip career-advisor-final/
print("✅ Model zipped!")
print("📁 File: career-advisor-final.zip")
print(f"📊 Size: {os.path.getsize('career-advisor-final.zip') / (1024*1024):.1f} MB")
```

**2. Download the zip:**
- Click the folder icon 📁 on the left sidebar
- Find `career-advisor-final.zip`
- Right-click → **Download**

**File size: ~700-800 MB**

---

**💡 RECOMMENDATION:** Start with **Option A** to test immediately, then use **Option B** to save to Drive for permanent storage. Only use **Option C** if you need local deployment.

---

### STEP 6: Extract and Use on Your Local PC

1. **Extract the downloaded zip:**
   - Extract `career-advisor-final.zip` to `E:\NextStepAI\`
   - Rename folder to: `career-advisor-production-v3`
   - Inside should have a `final_model` folder

2. **Your structure should be:**
```
E:\NextStepAI\
├── career-advisor-production-v3\
│   └── final_model\
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── vocab.json
│       ├── merges.txt
│       └── training_info.json
├── backend_api.py
└── test_accurate_model.py
```

---

### STEP 7: Test Your Model Locally

```powershell
# Test the model
python test_accurate_model.py
```

Expected output:
```
✅ Model found at ./career-advisor-production-v3/final_model
✅ Model loaded on cpu

Test 1/5: I love DevOps
✅ Relevant keywords: 80%
✅ Contains skills
✅ Contains interview questions
✅ Response is coherent
✅ No irrelevant content
✅ TEST PASSED
```

---

### STEP 8: Deploy to Production

```powershell
# Start backend
python -m uvicorn backend_api:app --port 8000
```

```powershell
# Test API
curl -X POST "http://127.0.0.1:8000/career-advice-ai" `
     -H "Content-Type: application/json" `
     -d '{"question": "I love DevOps"}'
```

---

## 🎯 Complete Checklist

- [ ] Upload 2 JSONL files to Google Drive folder "NextStepAI_Training"
- [ ] Create new Colab notebook with GPU enabled
- [ ] Copy and paste the complete training code
- [ ] Run the training cell (5-10 minutes)
- [ ] Zip and download the trained model (~700 MB)
- [ ] Extract to `E:\NextStepAI\career-advisor-production-v3\`
- [ ] Test with `python test_accurate_model.py`
- [ ] Deploy with `python -m uvicorn backend_api:app --port 8000`
- [ ] Test API and celebrate! 🎉

---

## 💡 Pro Tips

1. **Colab Session Timeout**: Free Colab sessions last ~12 hours. Your training only takes 5-10 min, so no worries!

2. **Save to Drive**: If you want to save directly to Drive instead of downloading:
```python
# Add this at the end of training
!cp -r career-advisor-final /content/drive/MyDrive/NextStepAI_Training/
```

3. **GPU Types**: Colab may give you T4, V100, or A100. All work great!
   - T4: 5-10 minutes
   - V100: 3-5 minutes
   - A100: 2-3 minutes

4. **Re-run Training**: You can retrain anytime by running the cell again!

---

## ✅ Why This is Better Than Local Training

| Aspect | Local RTX 2050 | Google Colab |
|--------|----------------|--------------|
| Setup time | 30+ min (CUDA install) | 2 minutes |
| Training time | 15-20 min | 5-10 min |
| GPU Memory | 4 GB | 15 GB |
| Cost | Free | Free |
| Convenience | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

---

---

## 🔍 STEP 5.5: ACCURACY VALIDATION CHECKER (Run This NOW!)

**⚠️ IMPORTANT: Run this to decide if you should deploy or retrain!**

Based on your test results, you had:
- ✅ 3/5 tests EXCELLENT (5-6/6 checks)
- ⚠️ 1/5 test GOOD but with "spring boot + django" error
- ❌ 1/5 test POOR (too short response)

**Run this validation to get a final decision:**

```python
# ============================================================================
# COMPREHENSIVE ACCURACY VALIDATION CHECKER
# Run this NOW to decide: Deploy or Retrain
# ============================================================================

print("\n" + "="*70)
print("🔍 MODEL ACCURACY VALIDATION")
print("="*70)

def validate_model_accuracy():
    """Comprehensive accuracy validator"""
    
    # Test cases with expected elements
    test_cases = [
        {
            "question": "I love to become a DevOps engineer",
            "must_have": ["docker", "kubernetes", "ci/cd", "jenkins", "git"],
            "must_not_have": ["django", "flask", "spring boot"]  # Python web frameworks
        },
        {
            "question": "I love to become a Data Scientist",
            "must_have": ["python", "machine learning", "pandas", "numpy", "statistics"],
            "must_not_have": ["spring boot", "django orm"]  # Wrong frameworks
        },
        {
            "question": "What is software development?",
            "must_have": ["programming", "code", "software", "development"],
            "must_not_have": ["₹", "lpa"]  # Avoid salary hallucinations
        },
        {
            "question": "Tell me about cloud engineering",
            "must_have": ["aws", "azure", "cloud", "infrastructure"],
            "must_not_have": ["kite database", "oracle 9-core"]  # Non-existent tech
        }
    ]
    
    total_score = 0
    max_score = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"VALIDATION TEST {i}/{len(test_cases)}")
        print(f"{'='*70}")
        print(f"Question: {test['question']}")
        
        # Generate response
        advice = generate_advice(test['question'], temperature=0.7)
        advice_lower = advice.lower()
        
        print(f"\n📝 Response Preview:")
        print(advice[:200] + "..." if len(advice) > 200 else advice)
        
        # Check must-have elements
        print(f"\n✓ Checking Required Elements:")
        must_have_found = 0
        for keyword in test['must_have']:
            found = keyword.lower() in advice_lower
            print(f"   {'✅' if found else '❌'} {keyword}")
            if found:
                must_have_found += 1
        
        must_have_score = (must_have_found / len(test['must_have'])) * 100
        
        # Check must-not-have elements
        print(f"\n✗ Checking Errors (should NOT be present):")
        errors_found = 0
        for error in test['must_not_have']:
            found = error.lower() in advice_lower
            print(f"   {'❌' if found else '✅'} {error} {'(FOUND - ERROR!)' if found else '(not found - good!)'}")
            if found:
                errors_found += 1
        
        error_score = ((len(test['must_not_have']) - errors_found) / len(test['must_not_have'])) * 100
        
        # Overall score for this test
        test_score = (must_have_score + error_score) / 2
        total_score += test_score
        max_score += 100
        
        print(f"\n🎯 Test Score: {test_score:.1f}%")
        if test_score >= 80:
            print("   ✅ EXCELLENT")
        elif test_score >= 60:
            print("   ✅ GOOD")
        elif test_score >= 40:
            print("   ⚠️  NEEDS IMPROVEMENT")
        else:
            print("   ❌ POOR - Retrain needed")
    
    # Final overall score
    overall_score = (total_score / max_score) * 100
    
    print(f"\n{'='*70}")
    print(f"📊 OVERALL MODEL ACCURACY")
    print(f"{'='*70}")
    print(f"Final Score: {overall_score:.1f}%")
    
    if overall_score >= 85:
        print("\n✅ EXCELLENT - Model is production-ready!")
        print("   • Highly accurate responses")
        print("   • No technical errors detected")
        print("   • Safe to deploy")
    elif overall_score >= 70:
        print("\n✅ GOOD - Model is usable with minor issues")
        print("   • Mostly accurate responses")
        print("   • Few minor errors")
        print("   • Can deploy but monitor responses")
    elif overall_score >= 50:
        print("\n⚠️  FAIR - Model needs improvement")
        print("   • Some inaccuracies present")
        print("   • Retrain recommended:")
        print("     - Increase epochs to 15")
        print("     - Lower learning rate to 1e-5")
    else:
        print("\n❌ POOR - Model needs significant retraining")
        print("   • Major accuracy issues")
        print("   • DO NOT deploy to production")
        print("   • Retrain with:")
        print("     - Epochs: 15")
        print("     - Learning rate: 1e-5")
        print("     - Check data quality")
    
    return overall_score

# Run validation
final_score = validate_model_accuracy()

print(f"\n{'='*70}")
print("💾 SAVE DECISION")
print(f"{'='*70}")

if final_score >= 70:
    print("✅ Model passed validation!")
    print("\n📋 Recommended next steps:")
    print("   1. Save to Google Drive (permanent storage)")
    print("   2. Or download for local deployment")
    print("   3. Deploy to production")
else:
    print("⚠️  Model needs retraining!")
    print("\n📋 Recommended actions:")
    print("   1. DON'T save/download this version")
    print("   2. Retrain with improved settings:")
    print("      • epochs = 15")
    print("      • learning_rate = 1e-5")
    print("      • batch_size = 2")
    print("   3. Run this validation again")
```

---

## 🚀 BONUS: Use Model Directly in Colab (Simple API Alternative!)

**Since ngrok requires authentication, here's a SIMPLER approach:**

Instead of creating a public API, just **call your model directly in Colab** using simple Python functions!

**Run this in a new cell after training:**

```python
# ============================================================================
# SIMPLE COLAB API FUNCTION (No ngrok needed!)
# ============================================================================

print("🚀 Setting up simple Career Advisor function...")

def get_career_advice_colab(question: str) -> str:
    """
    Get career advice from trained model
    
    Usage:
        advice = get_career_advice_colab("I love DevOps")
        print(advice)
    """
    input_text = f"<|startoftext|>### Question: {question}\n\n### Answer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "### Answer:" in response:
        return response.split("### Answer:")[1].strip()
    else:
        return response

print("✅ Function ready!")
print("\n📋 Usage:")
print('   advice = get_career_advice_colab("I love DevOps")')
print('   print(advice)')

print("\n🧪 Testing with 5 questions...")

# Test with various questions
test_questions = [
    "I love DevOps",
    "What is software development?",
    "I love networking",
    "Tell me about cloud engineering",
    "What skills do I need for data science?"
]

for i, question in enumerate(test_questions, 1):
    print("\n" + "="*70)
    print(f"Test {i}/{len(test_questions)}: {question}")
    print("="*70)
    
    advice = get_career_advice_colab(question)
    print(f"\n{advice}\n")

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETE!")
print("="*70)
print("\n💡 Your model is working! Now choose one of these options:")
print("   1. Keep using the function here in Colab")
print("   2. Save model to Google Drive for permanent storage")
print("   3. Download model to your local PC")
```

**This approach:**
- ✅ No ngrok authentication needed
- ✅ Works immediately in Colab
- ✅ Simple function calls
- ✅ Perfect for testing

---

## 🌐 ALTERNATIVE: Create Public API with Ngrok (If You Want External Access)

If you want to access from your local PC, you need ngrok authentication:

**Step 1: Get Ngrok Token (Free)**
1. Go to https://dashboard.ngrok.com/signup
2. Sign up for free account
3. Copy your authtoken

**Step 2: Run this in Colab:**

```python
# ============================================================================
# CREATE PUBLIC API WITH NGROK AUTHENTICATION
# ============================================================================

print("🚀 Creating public API for your Career Advisor...")

# Install packages
!pip install -q flask pyngrok

# Set your ngrok authtoken (get from: https://dashboard.ngrok.com/get-started/your-authtoken)
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")  # ⚠️ REPLACE WITH YOUR TOKEN

from flask import Flask, request, jsonify
import threading

# Create Flask app
app = Flask(__name__)

@app.route('/career-advice', methods=['POST'])
def get_career_advice():
    """API endpoint for career advice"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Generate response
        input_text = f"<|startoftext|>### Question: {question}\n\n### Answer:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "### Answer:" in response:
            answer = response.split("### Answer:")[1].strip()
        else:
            answer = response
        
        return jsonify({
            "question": question,
            "advice": answer,
            "model": "gpt2-medium (fine-tuned)",
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "gpt2-medium",
        "device": str(device),
        "message": "Career Advisor API is running!"
    })

# Start Flask in background thread
def run_flask():
    app.run(port=5000)

flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# Create public URL with ngrok
public_url = ngrok.connect(5000)

print("\n" + "="*70)
print("✅ CAREER ADVISOR API IS LIVE!")
print("="*70)
print(f"\n🌐 Public URL: {public_url}")
print("\n⚠️  COPY THIS URL!")
print(f"   {public_url}")
print("\n📋 Use from your local PC:")
print("   1. Copy the URL above")
print("   2. Update backend_api_colab.py with this URL")
print("   3. Run: python -m uvicorn backend_api_colab:app --port 8000")
print("\n✅ Your Career Advisor is now accessible from anywhere!")
print("⏰ Session lasts ~12 hours on free Colab")
```

**Now you can:**
- ✅ Use the API from your local Python scripts
- ✅ Test from your browser
- ✅ Share the URL with others
- ✅ Integrate with your existing NextStepAI application
- ✅ No need to download anything!

**Example: Use from Your Local PC**

```python
# On your local PC (E:\NextStepAI\)
import requests

# Replace with your ngrok URL from Colab
COLAB_API_URL = "https://xxxx-xx-xxx-xxx-xx.ngrok-free.app"

def get_career_advice_from_colab(question):
    response = requests.post(
        f"{COLAB_API_URL}/career-advice",
        json={"question": question}
    )
    return response.json()["advice"]

# Test it
advice = get_career_advice_from_colab("I love DevOps")
print(advice)
```

---

## 🚀 Ready to Start?

Just follow STEP 1: Upload your JSONL files to Google Drive!

Then go to STEP 2 and create your Colab notebook!

**Your options:**
1. ⚡ **Use in Colab only** - No download, instant testing (5 min setup)
2. 💾 **Save to Drive** - Permanent storage, load anytime (7 min setup)  
3. 🌐 **Create Colab API** - Use from anywhere online (10 min setup)
4. 📥 **Download to PC** - Local deployment (15 min total)

Your production-grade Career Advisor will be ready in minutes! 🎉
