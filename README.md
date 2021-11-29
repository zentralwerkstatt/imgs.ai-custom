# imgs.ai

**imgs.ai** is a fast, dataset-agnostic, deep visual search engine for digital art history based on neural network embeddings. It utilizes modern approximate k-NN algorithms via [Spotify's Annoy library](https://github.com/spotify/annoy) to deliver fast search results even for very large datasets in low-resource environments, and integrates the [OpenAI CLIP model](https://openai.com/blog/clip/) for text-based visual search. Try it [here](https://imgs.ai) on the complete [Rijksmuseum](https://www.rijksmuseum.nl) and [Metropolitan Museum of Art](https://www.metmuseum.org) collections or sign up for an account to access more functions/datasets (institutional email address and approval required.) imgs.ai is developed by [Fabian Offert](https://zentralwerkstatt.org), with contributions by Peter Bell and Oleg Harlamov. Get in touch at hi@imgs.ai.

**This repository provides a custom training function. It is independent of the main [imgs.ai repository](https://github.com/zentralwerkstatt.imgs.ai).**

## Local installation (experimental)

Only MacOS and Linux environments are currently supported.

1. Download and install the [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (preferred) package manager.
2. Create a Python 3.8 conda environment with `conda create --yes -n imgs.ai-custom python=3.8` and activate it with `conda activate imgs.ai-custom`.
3. Clone or download the repository and run the [install.sh](install.sh) shell script with your preferred shell. If you would like to install with GPU support (**GPU is strongly recommended**), add the following parameter: `cudatoolkit=10.1`, where the version number is the version of your installed CUDA framework (see https://pytorch.org/ for more information).
4. Edit [embedders.pytxt](embedders.pytxt) and the parameters of `make_model`in [train.py](train.py) to fit your needs, then run the script.

## CoLab version (experimental)

A CoLab version that allows GPU training in the cloud on data stored in Google Drive is available as a Jupyter notebook: [train.ipynb](train.ipynb). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zentralwerkstatt/imgs.ai-custom/blob/master/train.ipynb)