# deepcoder

## Create a virtual environment for python
If virtualenv is not installed, install it with the following command.
```
python -m pip install virtualenv
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
## Install python packages
We install the transformers, torch and the accelerate package. The latter one is needed to distribute our ai models to one or multiple GPUs (if available). For this tutorial it is important to install the Transformers package with a specific version because at the moment the DeepSeek model, which we will use, lacks compatibility with the latest transformers version.
```
pip install transformers==4.47.1 torch==2.6.0 accelerate==1.6.0
```
Moreover, let's install some packages for web development as we want to instruct our models to create some web services based on flask, jinja and wtForms.
```
pip install Jinja2 Flask WTForms
```


