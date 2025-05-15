from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# define the model
model_checkpoint = "Qwen/Qwen2-0.5B-Instruct"

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
   Create runnable python code to calculate the area
   of a circle and provide a flask endpoint on host 0.0.0.0 
   with the GET method for it.
   """

# tokenize the text prompt. Put the tokenized input
# on the same device as the model (cpu or gpu)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

# generate the output with the model
outputs = model.generate(**inputs, max_length=7000)

# decode the output with the tokenizer
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

