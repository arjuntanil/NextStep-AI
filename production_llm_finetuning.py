"""
Production-Grade LLM Fine-Tuning for Career Advisor
Uses DistilGPT-2 with LoRA for efficient training and deployment
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "distilgpt2"  # 82M parameters - fast and efficient
OUTPUT_DIR = "./career-advisor-production"
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

def load_training_data():
    """Load all knowledge base files for training"""
    data_files = [
        "E:/NextStepAI/career_advice_dataset.jsonl",
        "E:/NextStepAI/career_advice_ultra_clear_dataset.jsonl"
    ]
    
    training_examples = []
    for file_path in data_files:
        if os.path.exists(file_path):
            logger.info(f"✅ Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'prompt' in data and 'completion' in data:
                            # Format for causal LM: combine prompt and completion
                            text = f"### Question: {data['prompt']}\n\n### Answer: {data['completion']}<|endoftext|>"
                            training_examples.append({"text": text})
                    except Exception as e:
                        logger.warning(f"Skipping invalid line: {e}")
    
    logger.info(f"✅ Loaded {len(training_examples)} training examples")
    return training_examples

def prepare_dataset(examples, tokenizer):
    """Tokenize and prepare dataset for training"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
    
    dataset = Dataset.from_list(examples)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def setup_lora_model(model):
    """Configure LoRA for efficient fine-tuning"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn", "c_proj"],  # DistilGPT-2 attention layers
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train_model():
    """Main training function"""
    logger.info("🚀 Starting Production LLM Fine-Tuning")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Output: {OUTPUT_DIR}")
    
    # 1. Load tokenizer and model
    logger.info("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # 2. Apply LoRA for efficient training
    logger.info("Configuring LoRA...")
    model = setup_lora_model(model)
    
    # 3. Load and prepare training data
    logger.info("Loading training data...")
    training_examples = load_training_data()
    train_dataset = prepare_dataset(training_examples, tokenizer)
    
    # 4. Setup training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    # 5. Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 6. Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 7. Train the model
    logger.info("🔥 Starting training...")
    trainer.train()
    
    # 8. Save the fine-tuned model
    logger.info("💾 Saving fine-tuned model...")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    # 9. Save model info
    model_info = {
        "base_model": MODEL_NAME,
        "training_examples": len(training_examples),
        "epochs": NUM_EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "status": "production_ready"
    }
    
    with open(f"{OUTPUT_DIR}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info("✅ ✅ ✅ TRAINING COMPLETE ✅ ✅ ✅")
    logger.info(f"Model saved to: {OUTPUT_DIR}/final_model")
    logger.info("Ready for production deployment!")
    
    return trainer

def test_model():
    """Quick test of the fine-tuned model"""
    logger.info("\n🧪 Testing fine-tuned model...")
    
    tokenizer = AutoTokenizer.from_pretrained(f"{OUTPUT_DIR}/final_model")
    model = AutoModelForCausalLM.from_pretrained(
        f"{OUTPUT_DIR}/final_model",
        device_map="auto"
    )
    
    test_questions = [
        "I love DevOps",
        "Tell me about cloud engineering",
        "What skills do I need for data science?"
    ]
    
    for question in test_questions:
        logger.info(f"\n📝 Question: {question}")
        
        input_text = f"### Question: {question}\n\n### Answer:"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("### Answer:")[-1].strip()
        logger.info(f"💡 Answer: {answer[:200]}...")

if __name__ == "__main__":
    # Train the model
    trainer = train_model()
    
    # Test the model
    test_model()
    
    print("\n" + "="*60)
    print("✅ PRODUCTION LLM FINE-TUNING COMPLETE!")
    print(f"📁 Model location: {OUTPUT_DIR}/final_model")
    print("🚀 Ready to deploy in backend_api.py")
    print("="*60)
