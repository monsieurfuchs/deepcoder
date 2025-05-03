import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
from queue import Queue
import threading

app = Flask(__name__)

# Load the model outside the Flask server at the start
model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, max_length=5000, device_map='auto', torch_dtype=torch.bfloat16)

# Queue for holding inference jobs
job_queue = Queue()
# Dictionary to store results
results = {}
# Dictionary to keep track of job positions in the queue
job_positions = {}

def infer_job():
    while True:
        job_id, input_text = job_queue.get()
        if input_text is None:
            break
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, top_k=2)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results[job_id] = result
        job_queue.task_done()

# Start the inference thread
inference_thread = threading.Thread(target=infer_job)
inference_thread.daemon = True
inference_thread.start()

@app.route('/infer', methods=['POST'])
def infer():
    data = request.json
    input_text = data.get('text')
    if not input_text:
        return jsonify({'error': 'No text provided'}), 400
    
    job_id = len(job_positions)
    job_queue.put((job_id, input_text))
    job_positions[job_id] = job_queue.qsize()
    return jsonify({'job_id': job_id}), 200

@app.route('/result', methods=['GET'])
def result():
    job_id = request.args.get('job_id')
    if job_id is None:
        return jsonify({'error': 'No job_id provided'}), 400
    
    job_id = int(job_id)
    if job_id in results:
        return results[job_id]
    elif job_id in job_positions:
        return jsonify({'status': 'not ready', 'position': job_positions[job_id]}), 200
    else:
        return jsonify({'error': 'no such job'}), 404

@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>Welcome to the DeepSeek-Coder-V2-Lite-Instruct API</h1>
    <p>Endpoints:</p>
    <ul>
        <li><strong>/infer</strong>: POST request with JSON body containing "text" key for inference. Returns job ID.</li>
        <li><strong>/result</strong>: GET request with "job_id" parameter. Returns result if ready, otherwise status "not ready" and position in queue.</li>
    </ul>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

