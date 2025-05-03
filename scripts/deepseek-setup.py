from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

prompt = """
        Write python code to provide a flask endpoint with path /infer on port 5000 and host 0.0.0.0
        to the deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct model.
        Load the model outside the flask server at the start. Use trust_remote_code=True, max_length=5000, device_map='auto' and torch_dtype=torch.bfloat16.
    
        If a http requests comes in, put the job to infer on the model into a queue returning a counter as job id and process the jobs one after another.
        When the job is finished, store its inference output with the job id. Use top_k=2 for generating the model output.
        
        Provide a second endpoint under the path /result with the job id as parameter to return the respective inference result as plain text, not json.
        Remember all job ids. When the user asks for the result of a job id that exists but is not yet present in the results
        return 'status = not ready' and the current position of the job in the queue to the user. If there is no such id, return 'no such job'.
        
        Provide a third endpoint under the root path returning a short description and usage explanation of the endpoints."""

inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_length=5000)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
