"""
Inference Script for Fine-tuned Pythia-160M Career Advisor
Test and interact with your fine-tuned model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import json

class CareerAdvisorInference:
    """
    Inference class for the fine-tuned Pythia-160M Career Advisor model
    """
    
    def __init__(self, base_model_name, adapter_path):
        """
        Initialize the inference pipeline
        
        Args:
            base_model_name (str): Name of the base model
            adapter_path (str): Path to the fine-tuned LoRA adapters
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[INFO] Initializing inference pipeline on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the base model and fine-tuned adapters"""
        
        try:
            print(f"[INFO] Loading tokenizer from {self.base_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=True  # Use cached files
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"[INFO] Loading base model {self.base_model_name}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                device_map=None,
                trust_remote_code=True,
                local_files_only=True  # Use cached files
            )
            
            # Ensure model is on CPU
            base_model = base_model.to('cpu')
            
            print(f"[INFO] Loading fine-tuned adapters from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(
                base_model, 
                self.adapter_path,
                torch_dtype=torch.float32
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            print("[SUCCESS] Model loaded successfully!")
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"[INFO] Total parameters: {total_params:,}")
            print(f"[INFO] Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise e
    
    def format_prompt(self, question):
        """
        Format a question using the training prompt structure
        
        Args:
            question (str): The career-related question
            
        Returns:
            str: Formatted prompt ready for inference
        """
        return f"### Question:\n{question}\n\n### Answer:\n"
    
    def generate_response(self, question, max_new_tokens=200, temperature=0.7, top_p=0.9):
        """
        Generate a career advice response for a given question
        
        Args:
            question (str): The career-related question
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature (0.1-1.0)
            top_p (float): Top-p sampling parameter
            
        Returns:
            str: Generated career advice response
        """
        
        # Format the prompt
        prompt = self.format_prompt(question)
        
        # Tokenize the input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to device (CPU)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        print(f"[INFO] Generating response for: '{question[:50]}...'")
        print(f"[INFO] Input tokens: {inputs['input_ids'].shape[1]}")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode the full response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "### Answer:\n" in full_response:
            answer = full_response.split("### Answer:\n", 1)[1].strip()
            
            # Clean up the answer (remove any trailing incomplete sentences)
            sentences = answer.split('. ')
            if len(sentences) > 1 and not sentences[-1].endswith(('.', '!', '?', ':')):
                answer = '. '.join(sentences[:-1]) + '.'
            
            return answer
        else:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
    
    def interactive_mode(self):
        """
        Start interactive chat mode for career advice
        """
        print("\n" + "=" * 70)
        print("    🤖 AI CAREER ADVISOR - INTERACTIVE MODE")
        print("=" * 70)
        print("Ask me anything about careers, skills, certifications, or interviews!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")
        
        while True:
            try:
                # Get user input
                question = input("💼 Your Question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thank you for using AI Career Advisor! Good luck with your career journey!")
                    break
                
                if not question:
                    print("Please enter a question about careers.\n")
                    continue
                
                # Generate and display response
                print(f"\n🤖 AI Career Advisor:")
                response = self.generate_response(question)
                print(f"{response}\n")
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using AI Career Advisor!")
                break
            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print("Please try asking your question again.\n")

def run_test_queries(advisor):
    """
    Run predefined test queries to demonstrate the model's capabilities
    
    Args:
        advisor (CareerAdvisorInference): The inference pipeline instance
    """
    
    test_queries = [
        "What are the key skills required for a Data Scientist role in India?",
        "How can I transition from a software developer to a DevOps engineer?", 
        "What certifications are most valuable for cybersecurity professionals?",
        "What is the typical career path for a finance professional in banking?",
        "Which programming skills are most in-demand for frontend developers?",
        "How should I prepare for a machine learning engineer interview?",
        "What are the emerging opportunities for cloud engineers in 2024?",
        "What soft skills are important for product managers?"
    ]
    
    print("\n" + "=" * 70)
    print("    🧪 RUNNING TEST QUERIES")
    print("=" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[TEST {i}] {query}")
        print("-" * 50)
        
        try:
            response = advisor.generate_response(query, max_new_tokens=150)
            print(f"🤖 Response: {response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

def main():
    """
    Main function to run inference testing
    """
    
    # Configuration
    BASE_MODEL = "EleutherAI/pythia-160m-deduped"
    
    # Try different possible adapter paths
    possible_paths = [
        "./pythia-160m-career-advisor/final",
        "./career-advisor-finetuned/checkpoint-25",
        "./pythia-160m-career-advisor/checkpoint-100",
        "./career-advisor-finetuned/final_checkpoint"
    ]
    
    adapter_path = None
    for path in possible_paths:
        if os.path.exists(path):
            adapter_path = path
            print(f"[INFO] Found model at: {adapter_path}")
            break
    
    if not adapter_path:
        print("❌ [ERROR] No fine-tuned model found!")
        print("Available paths checked:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease run the fine-tuning script first: python pythia_finetuning_complete.py")
        return
    
    try:
        # Initialize the inference pipeline
        print(f"[INFO] Initializing AI Career Advisor with model: {BASE_MODEL}")
        advisor = CareerAdvisorInference(BASE_MODEL, adapter_path)
        
        # Run test queries
        run_test_queries(advisor)
        
        # Ask user if they want interactive mode
        print("\n" + "=" * 70)
        choice = input("Would you like to try interactive mode? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes']:
            advisor.interactive_mode()
        else:
            print("\n✅ Testing completed! Your AI Career Advisor is working perfectly!")
            
    except Exception as e:
        print(f"\n❌ [ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()