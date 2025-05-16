from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# define the model
model_checkpoint = "Qwen/Qwen2.5-Coder-7B-Instruct"

# load the tokenizer fitting the model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

# load the model
# use device_map='auto' to distribute the model across multiple
# GPUs if available, otherwise run on cpu
model = AutoModelForCausalLM.from_pretrained(
                             model_checkpoint,
                             trust_remote_code=True, 
                             torch_dtype=torch.bfloat16, 
                             device_map='auto')

# define a text prompt
prompt = """
   Write python code to provide a flask server on port 5000 and host 0.0.0.0                   
   Load the Qwen/Qwen2.5-Coder-7B-Instruct.
   Load the model outside the flask server at the start.                                                                                               
   Use trust_remote_code=True, device_map='auto' and torch_dtype=torch.bfloat16.                 

   Create an endpoind /inferto submit a text prompt the model.                            
   The result should be returned asi a plain text - not json.                                                
   In the HTTP header set Content-Type: text/plain
   Create an explicit lock on the model in order to guarantee that only one prompt is processed at a time.
   Use max_length=7000 for the generate method of the model.
   
   Create a form with a text area to submit a prompt to the model.
   Also include text inputs for the model parameters top_k, top_p, temperature and do_sample.
   Pass those parameters to the generate method of the model.
   Provide the endpoint for the form under the root path /. 
   Return the HTML directly as string in the response of the http request.
"""

# tokenize the text prompt. Put the tokenized input
# on the same device as the model (cpu or gpu)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

# generate the output with the model
outputs = model.generate(**inputs, do_sample=False, max_length=7000)

# decode the output with the tokenizer
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

