from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def infer(model_checkpoint=None, prompt=None, trust_remote_code=True):
    if not model_checkpoint:
        model_checkpoint = "microsoft/Phi-3-mini-128k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=trust_remote_code, torch_dtype=torch.bfloat16, device_map='auto')
    if not prompt:
        prompt = """
            Create a web page with HTML and JavaScript to calculate the area of a circle.
            Publish the web page with flask on port 5001. Embed the HTML and JavaScript
            into the python code of the flask server.
            """

    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=5000)
    result = tokenizer.decode(outputs[0])
    return result
