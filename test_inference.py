# test_inference.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_career_advice(prompt: str):
    base_model_name = "EleutherAI/pythia-160m-deduped"
    adapter_path = "./career-advisor-finetuned/final_checkpoint"

    print("Loading base model...")
    # Load the base model in half-precision for efficiency
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Loading fine-tuned LoRA adapters...")
    # Load the LoRA adapters on top of the base model
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Prepare the prompt in the same format used for training
    formatted_prompt = f"### Question:\n{prompt}\n\n### Answer:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating response...")
    # Generate the response
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response to only show the answer part
    answer = response.split("### Answer:\n")[-1]
    return answer

if __name__ == "__main__":
    user_prompt = "What are some good certifications for a Data Scientist?"
    print(f"\n--- User Prompt ---\n{user_prompt}")
    
    advice = get_career_advice(user_prompt)
    
    print(f"\n--- AI Career Advisor Response ---\n{advice}")