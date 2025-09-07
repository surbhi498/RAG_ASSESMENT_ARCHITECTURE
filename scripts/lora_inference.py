from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Resize embeddings if needed
base_model.resize_token_embeddings(len(tokenizer))

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "scripts/lora_distilgpt2", ignore_mismatched_sizes=True)

# Prepare input with attention mask
inputs = tokenizer("What is the capital of France?", return_tensors="pt", padding=True)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

prompt = "Answer the following question with only the factual answer:\nQ: What is the capital of France?\nA:"
inputs = tokenizer(prompt, return_tensors="pt")
output_ids = model.generate(inputs.input_ids, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

