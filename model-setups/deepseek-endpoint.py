from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Load the model outside the flask server at the start
model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, max_length=7000, device_map='auto', torch_dtype=torch.bfloat16)
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    model = None

@app.route('/')
def home():
    return '''
    <form action="/infer" method="post">
        <textarea name="prompt" rows="10" cols="50"></textarea><br>
        <input type="submit" value="Submit">
    </form>
    '''

@app.route('/infer', methods=['POST'])
def infer():
    if model is None:
        return "Model could not be loaded", 500
    prompt = request.form['prompt']
    if not prompt:
        return "Prompt is required", 400
    try:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=7000)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result, 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        logging.error(f"Error processing the prompt: {e}")
        return "An error occurred while processing your request", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
