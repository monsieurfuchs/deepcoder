from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True, device_map='auto')
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

prompt = """
    Write python code to provide a flask endpoint on port 5000
    to the deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct model.
    Load the model in the constructor.
    Use trust_remote_code=True, max_length=1000, device_map='auto' and torch_dtype=torch.bfloat16. 
    Provide only the runnable python code. Do not use indents.
    """
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=1000)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
