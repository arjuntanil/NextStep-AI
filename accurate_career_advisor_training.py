"""
Accurate Career Advisor LLM Fine-Tuning
Optimized for RTX 2050 GPU with improved parameters for accurate responses
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerAdvisorTrainer:
    """Trains GPT-2 for accurate career advice with skills and interview questions"""
    
    def __init__(self, model_name="gpt2", output_dir="./career-advisor-accurate"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.tokenizer = None
        self.model = None
        
    def load_training_data(self):
        """Load and prepare training data from JSONL files"""
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
                logger.info(f"✅ Loaded {file_path}: {len(all_examples)} examples so far")
        
        logger.info(f"📊 Total training examples: {len(all_examples)}")
        return all_examples
    
    def format_training_text(self, prompt, completion):
        """Format prompt and completion for better training"""
        # Use clear delimiters that help the model learn structure
        formatted = f"""<|startoftext|>### Question: {prompt}

### Answer: {completion}<|endoftext|>"""
        return formatted
    
    def prepare_dataset(self, examples):
        """Convert examples to tokenized dataset"""
        logger.info("🔧 Preparing dataset...")
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Format all text
        formatted_texts = [
            self.format_training_text(ex['prompt'], ex['completion']) 
            for ex in examples
        ]
        
        # Tokenize with proper truncation and padding
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding='max_length',
            max_length=512,  # Longer context for detailed responses
            return_tensors='pt'
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        })
        
        logger.info(f"✅ Dataset prepared: {len(dataset)} examples")
        return dataset
    
    def train_model(self, dataset):
        """Fine-tune GPT-2 with optimized parameters for RTX 2050"""
        logger.info("🚀 Starting fine-tuning...")
        
        # Load pre-trained model
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Optimized training arguments (works on CPU and GPU)
        use_cuda = torch.cuda.is_available()
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            
            # Training hyperparameters - optimized for accuracy
            num_train_epochs=5,  # Balanced epochs for CPU/GPU
            per_device_train_batch_size=2 if not use_cuda else 4,  # Smaller for CPU
            gradient_accumulation_steps=8 if not use_cuda else 4,  # Maintain effective batch size
            
            # Learning rate - optimized for convergence
            learning_rate=5e-5,  # Slightly higher for faster convergence
            warmup_steps=100,
            weight_decay=0.01,
            
            # Optimization
            fp16=use_cuda,  # Only use FP16 if CUDA available
            gradient_checkpointing=False,
            
            # Logging and saving
            logging_steps=20,
            save_steps=200,
            save_total_limit=2,
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            
            # Other
            remove_unused_columns=False,
            report_to="none",
            seed=42,
            dataloader_pin_memory=use_cuda  # Only pin memory if CUDA
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # GPT-2 is causal LM, not masked LM
        )
        
        # Split dataset for validation
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=split_dataset['train'],
            eval_dataset=split_dataset['test'],
            data_collator=data_collator,
        )
        
        # Train
        logger.info("🎯 Training started...")
        logger.info(f"   GPU: RTX 2050")
        logger.info(f"   Epochs: {training_args.num_train_epochs}")
        logger.info(f"   Batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"   Learning rate: {training_args.learning_rate}")
        
        trainer.train()
        
        logger.info("✅ Training completed!")
        return trainer
    
    def save_model(self):
        """Save the fine-tuned model"""
        final_model_path = self.output_dir / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving model to {final_model_path}...")
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        # Save training info
        info = {
            "base_model": self.model_name,
            "training_examples": "498 high-quality career examples",
            "model_type": "GPT-2 (124M parameters)",
            "optimization": "Optimized for RTX 2050 GPU",
            "capabilities": [
                "Generate skills for any job role",
                "Provide interview questions and answers",
                "Offer career guidance and certifications",
                "Structured and accurate responses"
            ]
        }
        
        with open(final_model_path / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info("✅ Model saved successfully!")
        return final_model_path
    
    def test_model(self, test_prompts):
        """Test the fine-tuned model"""
        logger.info("\n" + "="*60)
        logger.info("🧪 TESTING FINE-TUNED MODEL")
        logger.info("="*60)
        
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\n{'─'*60}")
            logger.info(f"Test {i}/{len(test_prompts)}: {prompt}")
            logger.info('─'*60)
            
            # Format input
            input_text = f"<|startoftext|>### Question: {prompt}\n\n### Answer:"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
            
            # Generate response
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
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode and clean response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            if "### Answer:" in full_response:
                answer = full_response.split("### Answer:")[1].strip()
            else:
                answer = full_response
            
            logger.info(f"\n💡 Response:\n{answer[:500]}...")
            
            # Check quality
            has_skills = any(word in answer.lower() for word in ['skill', 'skills', 'learn', 'knowledge'])
            has_questions = any(word in answer.lower() for word in ['question', 'interview', 'ask'])
            has_structure = '###' in answer or '*' in answer or '\n' in answer
            
            logger.info(f"\n📊 Quality Check:")
            logger.info(f"   {'✅' if has_skills else '❌'} Contains skills/learning content")
            logger.info(f"   {'✅' if has_questions else '❌'} Contains interview questions")
            logger.info(f"   {'✅' if has_structure else '❌'} Has structured formatting")
        
        logger.info("\n" + "="*60)
        logger.info("✅ TESTING COMPLETE")
        logger.info("="*60)


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("🎓 ACCURATE CAREER ADVISOR LLM TRAINING")
    print("   Optimized for RTX 2050 GPU")
    print("="*60 + "\n")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ No GPU detected, training will use CPU (slower)")
    
    # Initialize trainer
    trainer = CareerAdvisorTrainer(
        model_name="gpt2",  # GPT-2 (124M params) - good balance for RTX 2050
        output_dir="./career-advisor-accurate"
    )
    
    # Load data
    examples = trainer.load_training_data()
    
    if len(examples) == 0:
        print("❌ No training data found!")
        return
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(examples)
    
    # Train model
    trainer.train_model(dataset)
    
    # Save model
    model_path = trainer.save_model()
    
    # Test model
    test_prompts = [
        "I love DevOps",
        "Tell me about Software Development",
        "I love networking",
        "What skills do I need for Cloud Engineering?",
        "Tell me about Data Science career"
    ]
    
    trainer.test_model(test_prompts)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Model saved to: {model_path}")
    print("\n🚀 Next steps:")
    print("   1. Check the test results above")
    print("   2. The model will be automatically loaded by backend_api.py")
    print("   3. Start the server: python -m uvicorn backend_api:app --port 8000")


if __name__ == "__main__":
    main()
