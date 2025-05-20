from flask import Flask, request, render_template_string
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Lock
import sys

app = Flask(__name__)

# Load the model and tokenizer outside the Flask server
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Create a lock to ensure exclusive access to the model
lock = Lock()

@app.route('/', methods=['GET'])
def index():
    # HTML form to submit a prompt
    html_form = '''
    <form action="/infer" method="post">
        <textarea name="prompt" rows="4" cols="50"></textarea><br>
        Top-k: <input type="number" name="top_k" value="50"><br>
        Top-p: <input type="number" step="0.01" name="top_p" value="0.9"><br>
        Temperature: <input type="number" step="0.01" name="temperature" value="0.7"><br>
        Do Sample: <input type="checkbox" name="do_sample" checked><br>
        <input type="submit" value="Submit">
    </form>
    '''
    return html_form

@app.route('/infer', methods=['POST'])
def infer():
    with lock:
        prompt = request.form['prompt']
        top_k = int(request.form.get('top_k', 50))
        top_p = float(request.form.get('top_p', 0.9))
        temperature = float(request.form.get('temperature', 0.7))
        do_sample = bool(request.form.get('do_sample', False))

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_length=7000,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=do_sample,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    port = int(sys.argv[1])
    app.run(host='0.0.0.0', port=port)

