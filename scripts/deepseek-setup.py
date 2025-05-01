from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True, device_map='auto')
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

prompt = """
    Write python code to provide a flask endpoint on port 5000
    to the deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct model.
    Load the model in the constructor.

    If a http requests comes in, put the job to infer on the model into a queue returning a uniqie job id and process the jobs one after another.
    When the job is finished, store its inference output it with the job id.
    
    Provide a second endpoint with the job id as parameter to return the respective inference output.
    Remember all job ids. When the user asks for the result of job id that exists but is not yet present in the results
    return 'not ready' to the user. If there is no such id, return 'no such job'.
    Use trust_remote_code=True, max_length=1000, device_map='auto' and torch_dtype=torch.bfloat16. 

    Provide only the runnable python code without markups, comments or explanations.
    """
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=5000)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
