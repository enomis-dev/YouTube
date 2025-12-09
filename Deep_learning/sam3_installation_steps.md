1) Clone the repo

git clone https://github.com/facebookresearch/sam3.git
cd sam3

2) Then create an env to install sam3 dependencies

pyenv virtualenv 3.12 sam3
pyenv activate sam3

Install SAM3
pip install -e .

On my side I'll have to install specific cuda version in order to let pytorch to work on my gpu
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

3) 
Optionally you can install example notebook dependencies
pip install -e ".[notebooks]"

You can also optionally install development and training dependencies
pip install -e ".[train,dev]"

4) 
Then have to request sam 3 weights. The access so far need to be rquested on hugging face sam 3 page, once it is accepted you can download the files and put in the same repo of your project. In the notebook then you will reference the path 

https://huggingface.co/facebook/sam3/tree/main
