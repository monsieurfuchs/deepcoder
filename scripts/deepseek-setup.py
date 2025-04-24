from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16)

prompt = """
    Write python code to provide a flask endpoint on port 5000
    to the deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct model.
    Use trust_remote_code=True, max_length=1000 and torch_dtype=torch.bfloat16. 
    The result should be returned as plain text. Please give me only the python code without comments and explanations.
    """
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=1000)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
