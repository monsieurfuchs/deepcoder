# deepcoder

This short tutorial will introduce you into using sophisticated transformer models for natural language processing. The tutorial focuses on text-to-code generation utilizing generative ai to create runnable program code from human language instructions. The quality of code generation strongly depends on the respective model. By a rules of thumb larger models produce more reliable outputs and are more capable of working in a "self-employed" way - needing less detailed instructions to accomplish complex tasks. Moreover, models perform differently when combining tasks of different domains, such as producing code that contains both python and HTML/JavaScript code.

You will see that getting started with transformer models for mere inference is quite easy, requires only a few lines of code and only little prior knowledge, which is due to the transformers library being designed in a way that largely encapsulates all the complexity to load, deploy and infer on models.

## Prerequisites

We now start with the preparation of our python environment.

### Create a virtual environment for python
As it is common practice to cope with dependencies by using virtual environments let us create one. If virtualenv is not installed, install it with the following command.
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
We install the transformers, torch and the accelerate package. The latter one is needed to distribute our ai models to one or multiple GPUs (if available). For this tutorial it is important to install the Transformers package with a specific version because at the moment the DeepSeek model, which we will use, lacks compatibility with the latest transformers version. Normally this should not be the case, but wi will live with that because this tutorial is mailnly ment for educational purpose and we want to focus on the concepts of infering with language models.
```
pip install transformers==4.47.1 torch==2.6.0 accelerate==1.6.0
```
Moreover, let's install some packages for web development as we want to instruct our models to create some web services based on flask, jinja and wtForms.
```
pip install Jinja2 Flask WTForms
```

## Getting started with Transformers
Let's now create our first Transformers model for text-to-code generation. We will first use a rather small model that can be run on a low-resource system without GPU.

An important component of any NLP pipeline is the 'tokenizer'. A neural network does not understand words or sentences. Therefore sentences have to be split into 'tokens' which in most cases are words and assign them numbers, which a neural network can 'understand'. This is the task of the tokenizer which in general is bound to the respective model which means that a specific model is trained with a specific tokenizer that has to be used both for training the model and infering on it. For each input sequence the tokenizer produces a respective sequence of numbers that are fed into the network. The network then produces an output sequence of numbers conforming to the 'language' of the tokenizer. The tokenizer can then retranslate the network output into natural human language. 

Any processing pipeline consists of the same steps:

1. Initialize a tokenizer according to the model.
2. Initialize the model.
3. Create a prompt and have the tokenizer process it.
4. Feed the model with the output of the tokenizer - this will yield a sequence of numbers.
5. Again, use the tokenizer to translate the sequence of numbers into a sequence of words which is the final output.

First let's import the necessary packages.
```
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
```
Now let us define the model we want to use. For a first try, let's take the Phi-3-mini model with 3 Billion parameters which is pretty small and cpu-friendly. We identify the model by a 'checkpoint' string. The transformer package will then download the model for us. You can explore a huge variety of models and their checkpoints on huggingface.com. For example you can find the quen2 model for text-to-code generation here: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct 
```
model_checkpoint = "microsoft/Phi-3-mini-128k-instruct"
```
Initialize the tokenizer. The parameter trust_remote_code allows python to automatically download necessary stuff in the background. Having the transformers library download tokenizers, models and other code in the background is the usual way to work with that library. Once everything is downloaded and present on the system, the models etc. are persisted on the local machine and quickly loaded from there. The library functions first check if the models are available on the local machine otherwise they are downloaded from remote locations.
```
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
```
Initialize the model. The parameter setting device_map='auto' indicates that we want the model to be automatically distributed to GPUs if available. This is convinient for large models as they may be split across multiple GPUs in parallel.
```
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
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
Now the model has generated a tokenized output that we can decode to human-processable output.

```
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```



