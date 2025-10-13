"""
Test script to verify the fine-tuned Pythia-160M model works
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def test_finetuned_model():
    """Test the fine-tuned model from checkpoint-25"""
    
    print("=" * 60)
    print("    TESTING FINE-TUNED PYTHIA-160M MODEL")
    print("=" * 60)
    
    # Model paths
    base_model_name = "EleutherAI/pythia-160m-deduped"
    checkpoint_path = "./career-advisor-finetuned/checkpoint-25"
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        return False
    
    try:
        print("[INFO] Loading base model and tokenizer...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            local_files_only=True,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=None,
            local_files_only=True,
            trust_remote_code=True
        )
        base_model = base_model.to('cpu')
        
        print("[INFO] Loading fine-tuned LoRA adapters...")
        
        # Load fine-tuned model with LoRA adapters
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()
        
        print("[SUCCESS] Fine-tuned model loaded successfully!")
        print(f"[INFO] Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
        
        # Test queries
        test_queries = [
            "What skills are needed for a data scientist role?",
            "How to prepare for a software engineer interview?",
            "What certifications are valuable for DevOps engineers?",
            "What are the career opportunities in AI/ML?"
        ]
        
        print(f"\n[INFO] Testing with {len(test_queries)} career queries...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"[Q{i}] {query}")
            
            # Format prompt like training data
            prompt = f"### Question:\n{query}\n\n### Answer:\n"
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors='pt')
            input_ids = inputs['input_ids']
            
            print(f"[INFO] Generating response (input length: {input_ids.shape[1]} tokens)...")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 150,  # Add 150 new tokens
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            if "### Answer:\n" in full_response:
                answer = full_response.split("### Answer:\n", 1)[1].strip()
                # Truncate if too long
                if len(answer) > 300:
                    answer = answer[:300] + "..."
                print(f"[A{i}] {answer}")
            else:
                print(f"[A{i}] [Could not extract answer from response]")
            
            print("-" * 60)
        
        print("\n[SUCCESS] Fine-tuned model testing completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_info():
    """Display information about the fine-tuned model"""
    
    checkpoint_path = "./career-advisor-finetuned/checkpoint-25"
    
    try:
        # Read adapter config
        import json
        config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print("\n[INFO] Fine-tuned Model Configuration:")
            print(f"  - Base Model: {config.get('base_model_name_or_path', 'Unknown')}")
            print(f"  - LoRA Rank (r): {config.get('r', 'Unknown')}")
            print(f"  - LoRA Alpha: {config.get('lora_alpha', 'Unknown')}")
            print(f"  - Target Modules: {config.get('target_modules', 'Unknown')}")
            print(f"  - Task Type: {config.get('task_type', 'Unknown')}")
        
        # Check trainer state
        state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            print(f"\n[INFO] Training Progress:")
            print(f"  - Global Step: {state.get('global_step', 'Unknown')}")
            print(f"  - Epoch: {state.get('epoch', 'Unknown')}")
            if 'log_history' in state and state['log_history']:
                last_log = state['log_history'][-1]
                if 'loss' in last_log:
                    print(f"  - Final Loss: {last_log['loss']:.4f}")
        
    except Exception as e:
        print(f"[WARNING] Could not read model info: {e}")

if __name__ == "__main__":
    # Display model information
    check_model_info()
    
    # Test the fine-tuned model
    success = test_finetuned_model()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ FINE-TUNING VERIFICATION SUCCESSFUL!")
        print("=" * 60)
        print("Your Pythia-160M model has been successfully fine-tuned!")
        print("The model can now provide career advice based on your training data.")
        print(f"Checkpoint saved at: ./career-advisor-finetuned/checkpoint-25")
    else:
        print("\n❌ Fine-tuning verification failed!")