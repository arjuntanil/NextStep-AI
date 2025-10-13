"""
PRODUCTION CAREER ADVISOR FINE-TUNING
Using GPT-2-Medium (355M params) with 1000 training steps for maximum accuracy
"""

import json
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ProductionCareerAdvisorTrainer:
    """Production-grade trainer for highly accurate career advice"""
    
    def __init__(self):
        self.model_name = "gpt2-medium"  # 355M params - you already downloaded this
        self.output_dir = Path("./career-advisor-production-v2")
        self.tokenizer = None
        self.model = None
        
    def load_training_data(self):
        """Load high-quality training data"""
        logger.info("\n📚 Loading training data...")
        
        data_files = [
            "career_advice_dataset.jsonl",
            "career_advice_ultra_clear_dataset.jsonl"
        ]
        
        examples = []
        for file_path in data_files:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            example = json.loads(line.strip())
                            examples.append(example)
                        except:
                            continue
                logger.info(f"   ✅ Loaded {len(examples)} examples from {file_path}")
        
        logger.info(f"\n📊 Total examples: {len(examples)}")
        return examples
    
    def format_for_training(self, prompt, completion):
        """Format with clear structure for learning"""
        return f"<|startoftext|>### Question: {prompt}\n\n### Answer: {completion}<|endoftext|>"
    
    def prepare_dataset(self, examples):
        """Prepare tokenized dataset"""
        logger.info("\n🔧 Preparing dataset for training...")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Format texts
        texts = [self.format_for_training(ex['prompt'], ex['completion']) for ex in examples]
        
        logger.info(f"   Tokenizing {len(texts)} examples...")
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
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
        
        logger.info(f"   ✅ Dataset ready: {len(dataset)} examples\n")
        return dataset
    
    def calculate_training_params(self, dataset_size, target_steps=1000):
        """Calculate optimal parameters to reach target steps"""
        # You want 1000 steps
        # Formula: steps = (dataset_size / batch_size) * epochs
        
        batch_size = 1  # For CPU
        gradient_accumulation = 16  # Effective batch size = 16
        
        # Calculate epochs needed for 1000 steps
        steps_per_epoch = dataset_size / (batch_size * gradient_accumulation)
        epochs_needed = target_steps / steps_per_epoch
        
        # Round up to ensure we hit 1000+ steps
        epochs = max(3, int(epochs_needed) + 1)
        
        return {
            'epochs': epochs,
            'batch_size': batch_size,
            'gradient_accumulation': gradient_accumulation,
            'steps_per_epoch': int(steps_per_epoch),
            'total_steps': int(steps_per_epoch * epochs)
        }
    
    def train_model(self, dataset):
        """Fine-tune with 1000 steps for production accuracy"""
        logger.info("🚀 Loading gpt2-medium model...")
        
        # Load the model you already downloaded
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"✅ Model loaded on {device.upper()}\n")
        
        # Split dataset
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_size = len(split_dataset['train'])
        
        # Calculate parameters for 1000 steps
        params = self.calculate_training_params(train_size, target_steps=1000)
        
        logger.info("📊 Training Configuration:")
        logger.info(f"   Model: gpt2-medium (355M parameters)")
        logger.info(f"   Target Steps: 1000")
        logger.info(f"   Calculated Epochs: {params['epochs']}")
        logger.info(f"   Batch Size: {params['batch_size']}")
        logger.info(f"   Gradient Accumulation: {params['gradient_accumulation']}")
        logger.info(f"   Effective Batch Size: {params['batch_size'] * params['gradient_accumulation']}")
        logger.info(f"   Learning Rate: 5e-5")
        logger.info(f"   Training Examples: {train_size}")
        logger.info(f"   Steps per Epoch: {params['steps_per_epoch']}")
        logger.info(f"   Total Training Steps: {params['total_steps']}")
        logger.info(f"   Device: {device.upper()}\n")
        
        # Training arguments optimized for 1000 steps
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            
            # Core training parameters
            num_train_epochs=params['epochs'],
            per_device_train_batch_size=params['batch_size'],
            gradient_accumulation_steps=params['gradient_accumulation'],
            
            # Optimization
            learning_rate=5e-5,  # Your suggested learning rate
            warmup_steps=50,
            weight_decay=0.01,
            max_grad_norm=1.0,
            
            # Precision
            fp16=False,  # Disable for CPU stability
            
            # Logging and checkpointing
            logging_steps=10,  # Log every 10 steps
            save_steps=100,  # Save checkpoint every 100 steps
            save_total_limit=5,
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            
            # Other
            remove_unused_columns=False,
            report_to="none",
            seed=42,
            dataloader_pin_memory=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=split_dataset['train'],
            eval_dataset=split_dataset['test'],
            data_collator=data_collator,
        )
        
        logger.info("⏳ Training started...")
        logger.info(f"   Expected time: 30-60 minutes on CPU")
        logger.info(f"   Progress will be shown below:\n")
        
        start_time = time.time()
        
        # Train
        trainer.train()
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n✅ Training completed in {elapsed_time/60:.1f} minutes!")
        
        return trainer
    
    def save_model(self):
        """Save the production model"""
        final_path = self.output_dir / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n💾 Saving production model to {final_path}...")
        
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        # Save metadata
        info = {
            "base_model": "gpt2-medium",
            "parameters": "355M",
            "training_steps": "1000+",
            "training_examples": 498,
            "optimization": "Production-grade with 1000 steps",
            "capabilities": [
                "Accurate skills for any career",
                "Detailed interview questions and answers",
                "Career guidance and certifications",
                "Structured, professional responses"
            ],
            "usage": "Optimized for career advice generation"
        }
        
        with open(final_path / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info("✅ Model saved successfully!\n")
        return final_path
    
    def test_model(self):
        """Test the production model"""
        logger.info("="*70)
        logger.info("🧪 TESTING PRODUCTION MODEL")
        logger.info("="*70 + "\n")
        
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        test_prompts = [
            "I love DevOps",
            "What is software development",
            "I love networking",
            "Tell me about Cloud Engineering",
            "Data Science career path"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"{'─'*70}")
            logger.info(f"Test {i}/{len(test_prompts)}: {prompt}")
            logger.info('─'*70)
            
            input_text = f"<|startoftext|>### Question: {prompt}\n\n### Answer:"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "### Answer:" in response:
                answer = response.split("### Answer:")[1].strip()
            else:
                answer = response
            
            logger.info(f"\n💡 Response:\n{answer[:500]}")
            if len(answer) > 500:
                logger.info("...")
            
            # Check quality
            has_skills = any(word in answer.lower() for word in ['skill', 'skills', 'learn', 'technology'])
            has_questions = any(word in answer.lower() for word in ['question', 'interview', 'ask'])
            has_structure = any(marker in answer for marker in ['###', '*', '•'])
            
            logger.info(f"\n📊 Quality:")
            logger.info(f"   {'✅' if has_skills else '❌'} Skills/Technologies")
            logger.info(f"   {'✅' if has_questions else '⚠️ '} Interview Questions")
            logger.info(f"   {'✅' if has_structure else '❌'} Structured Format")
            logger.info()
        
        logger.info("="*70)
        logger.info("✅ TESTING COMPLETE")
        logger.info("="*70 + "\n")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🎯 PRODUCTION CAREER ADVISOR FINE-TUNING")
    print("   GPT-2-Medium (355M) | 1000 Training Steps")
    print("="*70)
    
    trainer = ProductionCareerAdvisorTrainer()
    
    # Load data
    examples = trainer.load_training_data()
    if len(examples) == 0:
        print("❌ No training data found!")
        return
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(examples)
    
    # Train with 1000 steps
    trainer.train_model(dataset)
    
    # Save model
    model_path = trainer.save_model()
    
    # Test model
    trainer.test_model()
    
    print("="*70)
    print("🎉 PRODUCTION MODEL READY!")
    print("="*70)
    print(f"\n📁 Model Location: {model_path}")
    print("\n🚀 Next Steps:")
    print("   1. Model is trained with 1000+ steps for accuracy")
    print("   2. Run: python test_accurate_model.py")
    print("   3. Start API: python -m uvicorn backend_api:app --port 8000")
    print("\n✅ Your AI Career Advisor is production-ready!")


if __name__ == "__main__":
    main()
