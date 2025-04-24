from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

class ModelHandler:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True, max_length=1000, device_map='auto', torch_dtype=torch.bfloat16)

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=1000)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

model_handler = ModelHandler()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    response = model_handler.generate_response(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(port=5000)
