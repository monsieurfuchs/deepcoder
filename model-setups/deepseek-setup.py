from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

prompt = """
        Write python code to provide a flask server on port 5000 and host 0.0.0.0
        Load the deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct model.
        Load the model outside the flask server at the start.
        Use trust_remote_code=True, max_length=7000, device_map='auto' and torch_dtype=torch.bfloat16.

        Under the endpoind /infer create an endpoint to send a text prompt to the model.
        The result should be returned asi a plain text - not json. in the HTTP header set Content-Type: text/plain
        Create an explicit lock on the model in order to guarantee that only one prompt is processed at a time.

        Create an html form with a text area to submit a prompt to the /infer endpoint.
        Provide the form under the root path / of the server and load the html as string directly into the python code of the flask server.

        Create another endpoint under /info providing some explanation about how transformer models work.
        """


inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_length=7000, top_k=1)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
