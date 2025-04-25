import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import queue
import threading
import uuid

app = Flask(__name__)
q = queue.Queue()
results = {}
job_ids = set()

def load_model():
    model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, max_length=1000, device_map='auto', torch_dtype=torch.bfloat16)
    return model, tokenizer

model, tokenizer = load_model()

def process_job():
    while True:
        job = q.get()
        if job is None:
            break
        input_text = job['input_text']
        job_id = job['job_id']
        with torch.no_grad():
            inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs, max_length=1000)
            result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results[job_id] = result_text
        q.task_done()

threading.Thread(target=process_job, daemon=True).start()

@app.route('/infer', methods=['POST'])
def infer():
    data = request.json
    input_text = data.get('input_text')
    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400
    job_id = str(uuid.uuid4())
    job_ids.add(job_id)
    q.put({'input_text': input_text, 'job_id': job_id})
    return jsonify({'job_id': job_id})

@app.route('/result/<job_id>', methods=['GET'])
def result(job_id):
    if job_id in results:
        return jsonify({'result': results[job_id]})
    elif job_id in job_ids:
        return 'not ready', 202
    else:
        return 'no such job', 404

if __name__ == '__main__':
    app.run(port=5000)
