import sys
from flask import Flask, request, render_template_string
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Lock

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

lock = Lock()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        top_k = int(request.form.get('top_k', 50))
        top_p = float(request.form.get('top_p', 0.9))
        temperature = float(request.form.get('temperature', 0.7))
        do_sample = bool(request.form.get('do_sample', False))

        with lock:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_length=7000, top_k=top_k, top_p=top_p, temperature=temperature, do_sample=do_sample)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text, {'Content-Type': 'text/plain'}
    
    # Render the HTML form
    html_form = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Qwen Model Prompt</title>
        <style>
            body { display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            form { text-align: center; }
        </style>
    </head>
    <body>
        <form action="/" method="post">
            <h1>Submit a Prompt to Qwen Model</h1>
            <textarea name="prompt" rows="4" cols="50"></textarea><br><br>
            Top K: <input type="number" name="top_k" value="50"><br><br>
            Top P: <input type="number" step="0.01" name="top_p" value="0.9"><br><br>
            Temperature: <input type="number" step="0.01" name="temperature" value="0.7"><br><br>
            Do Sample: <input type="checkbox" name="do_sample" checked><br><br>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """
    return html_form

if __name__ == '__main__':
    port = int(sys.argv[1])
    app.run(host='0.0.0.0', port=port)

