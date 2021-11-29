conda install --yes -c pytorch pytorch torchvision torchaudio ${1:-cpuonly}
conda install --yes requests tqdm scikit-learn h5py dill
pip install annoy umap-learn
pip install git+https://github.com/openai/CLIP.git