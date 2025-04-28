# deepcoder

## Prerequisites

### Create a virtual environment for python
If virtualenv is not installed, install it with the following command.
```
python -m pip install virtualenv
```
Some environments do not allow the direct use of pip and return an error like this:
```
error: externally-managed-environment
```
In this case follow the instructions provided with that error. For example in Ubuntu you will be adbviced to install with apt:
```
apt install python3-virtualenv
```

Now create a virtual environment with a name of your choice. Here we choose "ai-coder" 
```
python -m venv ai-coder
```
This creates a directory with the name "ai-coder". Navigate into that directory and activate the virtual environment.
```
cd ai-coder
source ./bin/activate
```
### Install python packages
We install the transformers, torch and the accelerate package. The latter one is needed to distribute our ai models to one or multiple GPUs (if available). For this tutorial it is important to install the Transformers package with a specific version because at the moment the DeepSeek model, which we will use, lacks compatibility with the latest transformers version.
```
pip install transformers==4.47.1 torch==2.6.0 accelerate==1.6.0
```
Moreover, let's install some packages for web development as we want to instruct our models to create some web services based on flask, jinja and wtForms.
```
pip install Jinja2 Flask WTForms
```

## Getting started with Transformers
Let's now create our first Transformers model for text-to-code generation. We will first use a rather small model that can be run on a low-resource system without GPU.

Any processing pipeline consists of the same steps:

1. Initialize a tokenizer according to the model in order to split natural language into small pieces like words and numbers to represent them.
2. Initialize the model.
3. Create a prompt and have the tokenizer process it.
4. Feed the model with the output of the tokenizer - this will yield a sequence of numbers.
5. Again, use the tokenizer to translate the sequence of numbers into a sequence of words which is the final output.

First let's import the necessary packages.

```
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
```

Now let us define the model we want to use. For a first try, let's take the quen2 with 1.5 Billion parameters model which is pretty small and cpu-friendly. We identify the model by a 'checkpoint' string. The transformer package will then download the model for us. You can explore a huge variety of models and their checkpoints on huggingface.com. For example you can find the quen2 model for text-to-code generation here: https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
```
model_checkpoint = "Qwen/Qwen2-1.5B-Instruct"
```

Initialize the tokenizer. 
```
tokenizer = AutoTokenizer.from_pretrained("model_checkpoint", trust_remote_code=True)
```

Initialize the model. The parameter setting device_map='auto' indicates that we want the model to be automatically distributed to GPUs if available. This is convinient for large models as they may be split across multiple GPUs in parallel.
```
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
```

Now we are ready to use the model and formulate a prompt: 

```
prompt = """
   Create a web page with HTML and JavaScript to calculate the area of a circle.
   Publish the web page with flask on port 5001. Embed the HTML and JavaScript
   into the python code of the flask server.
   """     
```
We tokenize the prompt and feed the tokenization result into the model.

```
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=1000)
```

Now the model has generated a tokenized output that we have to detokenize to human-processable output.

```
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```


