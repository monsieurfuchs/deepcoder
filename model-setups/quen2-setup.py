from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", trust_remote_code=True, device_map='auto')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

prompt = """
    Write python code to provide a flask endpoint on port 5000
    to the Qwen/Qwen2-1.5B-Instruct.
    Load the model in the constructor.
    If a http requests comes in, put the job to infer on the model into a queue returning a uniqie job id and process the jobs one after another.
    When the job is finished, store its inference output it with the job id.
    Provide a second endpoint with the job id as parameter to return the respective inference output.
    Remember all job ids. When the user asks for the result of job id that exists but is not yet present in the results
    return '\n not ready \n' to the user. If there is no such id, return '\n no such job \n'.
    Use trust_remote_code=True, max_length=1000, device_map='auto' and torch_dtype=torch.bfloat16. 

    Provide only the runnable python code. Do not use indents.
    """
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=1000)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
