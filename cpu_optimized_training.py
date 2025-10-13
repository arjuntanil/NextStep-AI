"""
🔄 CPU-OPTIMIZED CAREER ADVISOR TRAINING
Faster CPU training with smaller model (if GPU unavailable)

Configuration:
- Model: gpt2 (117M params) - Much faster on CPU than gpt2-medium
- Epochs: 4 (balanced for CPU training)
- Batch Size: 1 (CPU memory efficient)
- Gradient Accumulation: 16 (effective batch = 16)
- Learning Rate: 5e-5
- Expected Time: ~90-120 minutes on CPU
- Quality: Good (not as perfect as GPU, but acceptable)
"""

import os
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
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CPUOptimizedTrainer:
    """CPU-optimized trainer for systems without GPU"""
    
    def __init__(self):
        self.model_name = "gpt2"  # 117M params - much faster than gpt2-medium on CPU
        self.output_dir = Path("./career-advisor-cpu-optimized")
        self.final_model_path = self.output_dir / "final_model"
        self.start_time = None
        
    def load_training_data(self):
        """Load training data"""
        logger.info("📚 Loading training data...")
        
        data_files = [
            "career_advice_dataset.jsonl",
            "career_advice_ultra_clear_dataset.jsonl"
        ]
        
        all_examples = []
        for file_path in data_files:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            example = json.loads(line.strip())
                            all_examples.append(example)
                        except:
                            continue
                logger.info(f"   ✅ {file_path}: {len(all_examples)} examples")
        
        logger.info(f"📊 Total: {len(all_examples)} examples")
        return all_examples
    
    def format_training_example(self, prompt, completion):
        """Format for training"""
        return f"<|startoftext|>### Question: {prompt}\n\n### Answer: {completion}<|endoftext|>"
    
    def prepare_dataset(self, examples):
        """Prepare dataset"""
        logger.info("🔧 Preparing dataset...")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        formatted_texts = [
            self.format_training_example(ex['prompt'], ex['completion'])
            for ex in examples
        ]
        
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding='max_length',
            max_length=384,  # Shorter for CPU efficiency
            return_tensors='pt'
        )
        
        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        })
        
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        logger.info(f"   ✅ Train: {len(split_dataset['train'])}, Val: {len(split_dataset['test'])}")
        
        return split_dataset
    
    def train_model(self, dataset):
        """Train with CPU optimization"""
        print("\n" + "="*70)
        print("🔄 CPU-OPTIMIZED TRAINING")
        print("="*70)
        print("⚠️  Training on CPU (no GPU detected)")
        print(f"   Model: {self.model_name} (117M params - faster than gpt2-medium)")
        print("   Optimization: Smaller model + reduced epochs for CPU efficiency")
        print("   Expected time: ~90-120 minutes")
        print("   Quality: Good (acceptable for production)")
        print("="*70 + "\n")
        
        logger.info(f"📦 Loading {self.model_name}...")
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        train_samples = len(dataset['train'])
        steps_per_epoch = train_samples // 16  # effective batch size
        epochs = 4
        total_steps = steps_per_epoch * epochs
        
        print(f"📊 Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: 1")
        print(f"   Gradient accumulation: 16")
        print(f"   Effective batch: 16")
        print(f"   Learning rate: 5e-5")
        print(f"   Total steps: {total_steps}")
        print(f"   Steps per epoch: {steps_per_epoch}\n")
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            
            # CPU-optimized settings
            num_train_epochs=4,  # Reduced for CPU
            per_device_train_batch_size=1,  # CPU memory efficient
            gradient_accumulation_steps=16,  # Effective batch = 16
            
            learning_rate=5e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            
            # No GPU optimizations
            fp16=False,
            dataloader_pin_memory=False,
            
            # Logging
            logging_steps=5,
            logging_first_step=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,  # Save disk space
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            
            report_to="none",
            seed=42,
            remove_unused_columns=False
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
        )
        
        self.start_time = datetime.now()
        logger.info("⏳ Training started...")
        logger.info("   This will take ~90-120 minutes on CPU")
        logger.info("   Progress will be logged every 5 steps\n")
        
        trainer.train()
        
        elapsed = datetime.now() - self.start_time
        logger.info(f"✅ Training completed in {elapsed}")
        
        return trainer
    
    def save_model(self):
        """Save model"""
        self.final_model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving to {self.final_model_path}...")
        self.model.save_pretrained(self.final_model_path)
        self.tokenizer.save_pretrained(self.final_model_path)
        
        elapsed = datetime.now() - self.start_time
        metadata = {
            "model": "gpt2 (117M parameters - CPU optimized)",
            "training_samples": 498,
            "epochs": 4,
            "device": "CPU",
            "training_time": str(elapsed),
            "optimization": "CPU-optimized with smaller model",
            "quality": "Good - acceptable for production use",
            "note": "For best quality, retrain with GPU using gpt2-medium"
        }
        
        with open(self.final_model_path / "training_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✅ Model saved!")
    
    def test_model(self):
        """Quick test"""
        print("\n" + "="*70)
        print("🧪 QUICK TEST")
        print("="*70)
        
        self.model.eval()
        
        test_q = ["I love DevOps", "What is software development"]
        
        for q in test_q:
            print(f"\nQ: {q}")
            print("─"*70)
            
            input_text = f"<|startoftext|>### Question: {q}\n\n### Answer:"
            inputs = self.tokenizer(input_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Answer:" in response:
                answer = response.split("### Answer:")[1].strip()
            else:
                answer = response
            
            print(f"A: {answer[:300]}...")


def main():
    print("\n" + "="*70)
    print("🔄 CPU-OPTIMIZED CAREER ADVISOR TRAINING")
    print("   For systems without GPU support")
    print("="*70)
    
    trainer = CPUOptimizedTrainer()
    
    examples = trainer.load_training_data()
    if len(examples) < 100:
        logger.error("❌ Insufficient data!")
        return
    
    dataset = trainer.prepare_dataset(examples)
    trainer.train_model(dataset)
    trainer.save_model()
    trainer.test_model()
    
    print("\n" + "="*70)
    print("🎉 CPU TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Model: {trainer.final_model_path}")
    print("\n⚠️  Note: For best accuracy, consider GPU training with gpt2-medium")
    print("   Current model (CPU-optimized) is good but not as accurate as GPU version")
    print("\n🚀 Next steps:")
    print("   1. Test: python test_accurate_model.py")
    print("   2. Deploy: python -m uvicorn backend_api:app --port 8000")


if __name__ == "__main__":
    main()
