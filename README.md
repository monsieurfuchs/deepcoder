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
Now let us define the model we want to use. For a first try, let's take the Quen2 instruct model with only 500 Million parameters which is pretty small and cpu-friendly. We identify the model by a 'checkpoint' string. The transformer package will then download the model for us. You can explore a huge variety of models and their checkpoints on huggingface.com. For example you can find the quen2 model for text-to-code generation here: https://huggingface.co/Qwen/Qwen2-0.5B-Instruct 
```
model_checkpoint = "Qwen/Qwen2-0.5B-Instruct"
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
   Create runnable python code to calculate the area
   of a circle and provide a flask endpoint for it
   """
```

We tokenize the prompt and feed the tokenization result into the model.
```
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=5000)
```
Now the model has generated a tokenized output that we can decode to human-processable output.

```
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

Try the same procedure with a more advanced prompt and let's see how this small model performs:

```
prompt = """
   Create a web page with HTML and JavaScript to calculate the area of a circle.
   Publish the web page with flask on port 5001. Embed the HTML and JavaScript
   into the python code of the flask server.
   """
```

## top_k, top_p and temperature
If you run the previous prompt multiple times you will notice that the model generates different outputs. Normally you wouldn't expect this because once a neural network is trained its weights are fixed and we should expect the same output for the same input, shouldn't we? The reason for this benaviour is that the designers of the model wanted it to act this way to make the model apper more "creative" or human-like. We can influence that behavior with three parameters:

* top_k tells the model to consider only the top k next tokens ordered by their highest probability
* top_p is more inclusive than top_k and considers all tokens that sum up to a cummulatitive propability above some threshold
* temperature is a scaling factor to squeeze or broaden the probability distribution among the tokens. by this small probabilities may be punished or lifted.

For example - if you choose top_k=1, the modell is likely to process thes same output over multiple inferences on the same input.

## More advanced models
The small Quen2 model often does not provide functional and instantly runnable code. It makes mistakes, forgets things and seems to struggle with complex tasks - espcially when combining two or multiple tasks from different domains - like publishing HTML and JavaScript within a pythonic flask server. Let us therefore now try a much larger model - namely the Deepseek Coder model by repeating the above steps but with another checkpoint:

```
deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
```
Note that this is the light version of the model with 15.7 billion parameters. The full model (deepseek-ai/DeepSeek-Coder-V2-Instruct) has even 236 Billion parameters and eats up about 500 GB.


