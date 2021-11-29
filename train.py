# Filter annyoing PyTorch warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import PIL.Image
import torch as t
from uuid import uuid4
import os
import h5py
from io import BytesIO
import requests
import time
import torchvision as tv
import torch.nn as nn
import clip
from tqdm import tqdm
import csv
from threading import Lock, Thread
from queue import Queue, Empty
import dill as pickle # https://discuss.pytorch.org/t/cant-pickle-local-object-dataloader-init-locals-lambda/31857/32
import json
import random
from annoy import AnnoyIndex
import logging
from datetime import date
from sklearn.decomposition import PCA, IncrementalPCA


# Logging
logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(name)s : %(message)s")
log.info("Succesfully set up logging")


def img_from_url(url, max_tries=10):
    tries = 0
    while tries < max_tries:
        try:
            response = requests.get(url, timeout=30)
            img_bytes = BytesIO(response.content)
            img = PIL.Image.open(img_bytes).convert("RGB")
            return img
        except:
            tries += 1
        time.sleep(1)


def save_imgs_to(imgs, folder):    
    new_dir(folder)
    paths = []
    idxs = []
    for img in imgs:
        if isinstance(img, str): # URL or file path
            if img.startswith("http"):
                img = img_from_url(img)
            else:
                img = PIL.Image.open(img).convert("RGB")
        else: # Data stream
            stream = BytesIO(img.read())
            img = PIL.Image.open(stream).convert("RGB")
        idx = str(uuid4())
        path = str(os.path.join(folder, f"{idx}.jpg"))
        img.save(path)
        paths.append(path)
        idxs.append(idx)
    return paths, idxs


def load_img(path):
    return PIL.Image.open(path).convert("RGB")


def sort_dict(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}


def from_device(tensor):
    return tensor.detach().cpu().numpy()


def new_dir(folder):
    os.makedirs(folder, exist_ok=True)
    return folder


def arrange_data(X, shuffle=True, max_data=0):
    if shuffle: random.shuffle(X)
    if max_data: X = X[:max_data]
    return X


def set_cuda():
    device = "cuda" if t.cuda.is_available() else "cpu"
    return device


class Embedder_Raw:

    model = None

    def __init__(self, resolution=32, reducer=None, metrics=["manhattan"]):
        self.resolution = resolution
        self.feature_length = self.resolution * self.resolution * 3
        self.reducer = reducer
        self.metrics = metrics

    def transform(self, img, device="cpu"):
        img = img.resize((self.resolution, self.resolution), PIL.Image.ANTIALIAS)
        output = np.array(img)
        return output.astype(np.uint8).flatten()


class Embedder_VGG19:

    feature_length = 4096
    model = None
    
    def __init__(self, reducer=None, metrics=["manhattan"]):
        self.reducer = reducer
        self.metrics = metrics

    def transform(self, img, device="cpu"):
        if self.model is None:
            # Construct model only on demand
            self.model = tv.models.vgg19(pretrained=True).to(device)
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier.children())[:5]
            )  # VGG19 fc1
            self.model.eval()
            self.transforms = tv.transforms.Compose(
                [tv.transforms.Resize((224, 224)), tv.transforms.ToTensor()]
            )

        with t.no_grad():
            output = self.model(self.transforms(img).unsqueeze(0).to(device))
            return from_device(output).astype(np.float32).flatten()


class Embedder_CLIP_ViT:

    feature_length = 512
    model = None

    def __init__(self, reducer=None, metrics=["angular"]):
        self.reducer = reducer
        self.metrics = metrics

    def transform(self, img, device="cpu"):
        if self.model is None:
            self.model, self.transforms = clip.load("ViT-B/32", device=device)
            self.model.eval()

        with t.no_grad():
            input_ = self.transforms(img).unsqueeze(0).to(device)
            output = self.model.encode_image(input_)
            output /= output.norm(dim=-1, keepdim=True)
            return from_device(output).astype(np.float32).flatten()


class Embedder_Poses:

    feature_length = 17 * 2
    model = None

    def __init__(self, min_score=0.9, reducer=None, metrics=["manhattan", "angular", "euclidean"]):
        self.min_score = min_score
        self.reducer = reducer
        self.metrics = metrics

    def _normalize_keypoints(self, keypoints, scores):
        keypoints_scaled = np.zeros(self.feature_length) # Return empty array if no poses found
        if keypoints.shape[0] > 0:
            keypoints = keypoints[0]  # Already ranked by score
            score = scores[0].item()
            if self.min_score is None or score > self.min_score:
                # Scale w.r.t exact bounding box
                min_x = min([keypoint[0] for keypoint in keypoints])
                max_x = max([keypoint[0] for keypoint in keypoints])
                min_y = min([keypoint[1] for keypoint in keypoints])
                max_y = max([keypoint[1] for keypoint in keypoints])
                keypoints_scaled = []
                for keypoint in keypoints:
                    scaled_x = (keypoint[0] - min_x) / (max_x - min_x)
                    scaled_y = (keypoint[1] - min_y) / (max_y - min_y)
                    keypoints_scaled.extend([scaled_x, scaled_y])
                keypoints_scaled = np.array(keypoints_scaled)
        return keypoints_scaled

    def transform(self, img, device="cpu"):
        if self.model is None:
            self.model = tv.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device)
            self.model.eval()
            self.transforms = tv.transforms.Compose([tv.transforms.Resize(256), tv.transforms.ToTensor()])

        with t.no_grad():
            output = self.model(self.transforms(img).unsqueeze(0).to(device))
            scores = from_device(output[0]["scores"])
            keypoints = from_device(output[0]["keypoints"])
            normalized_keypoints = self._normalize_keypoints(keypoints, scores)
            return normalized_keypoints.astype(np.float32).flatten()


def collect_embed(X, embedders, data_root, num_workers, embs_file, start, end):
    device = set_cuda()

    X_dict = {}
    for i, x in enumerate(X):
        X_dict[i] = x

    if end is None:
        end = len(X_dict)
    
    # Allocate space
    log.info("Allocating space")
    embs = h5py.File(embs_file, "a")
    valid_idxs = []
    if start == 0:
        for emb_type, embedder in embedders.items():
            embs.create_dataset(emb_type.lower(), 
                                compression="lzf", 
                                shape=(len(X), embedder.feature_length))
    else:
        valid_idxs.extend(list(embs["valid_idxs"][:]))
        del embs["valid_idxs"]

    # Set up threading
    pbar_success = tqdm(total=(end-start)+1, desc="Embedded")
    pbar_failure = tqdm(total=(end-start)+1, desc="Failed")
    q = Queue()
    l = Lock()
    
    # Define and start queue
    def _worker():
        while True:
            try:
                i, x = q.get()
            except Empty:
                break
            path = x[0]
            success = False
            try:
                if path.startswith("http"):
                    img = img_from_url(path)
                else:
                    img = load_img(os.path.join(data_root, path))
            except:
                img = None
            if img:
                with l:
                    for emb_type, embedder in embedders.items():
                        embs[emb_type.lower()][i] = embedder.transform(img, device)
                    valid_idxs.append(i)
                    success = True
            with l:
                if success:
                    pbar_success.update(1)
                else:
                    pbar_failure.update(1)
            q.task_done()

    for i in range(num_workers):
        t = Thread(target=_worker)
        t.daemon = True
        t.start()

    for i,x in X_dict.items():
        if i>=start and i<=end:
            q.put((i, x))

    # Cleanup
    q.join()
    pbar_success.close()
    pbar_failure.close()
    embs.create_dataset("valid_idxs", compression="lzf", data=np.array(valid_idxs))
    embs.close()


def train(X, model_folder, embedders, data_root, num_workers, start, end, n_trees, build=True):
    # Set up
    log.info(f'Setting up config')
    config = {}
    config["data_root"] = data_root

    # Create or load raw embeddings
    embs_file = os.path.join(model_folder, "embeddings.hdf5")
    collect_embed(X, embedders, data_root, num_workers, embs_file, start, end)

    if build:

        embs = h5py.File(embs_file, "r")
        valid_idxs = list(embs["valid_idxs"])
        config["model_len"] = len(valid_idxs)

        # Allocate cache
        log.info(f'Allocating cache')
        cache_file = os.path.join(model_folder, "cache.hdf5")
        cache = h5py.File(cache_file, "w")

        # Reduce if reducer given
        log.info(f'Applying dimensionality reduction')
        for emb_type, embedder in embedders.items():
            data = None
            if embedder.reducer:
                log.info(embedder.reducer) ### DEBUG ###
                
                """
                ### DEBUG ###
                reducer = PCA(n_components=50, svd_solver='full') 
                log.info(embs[emb_type.lower()].shape) 
                log.info(embs[emb_type.lower()]) 
                log.info(np.isnan(embs[emb_type.lower()]).any())
                log.info(np.isinf(embs[emb_type.lower()]).any())
                data = reducer.fit_transform(embs[emb_type.lower()]) 
                ### DEBUG ###
                """

                data = embedder.reducer.fit_transform(embs[emb_type.lower()]) 
            else:
                data = embs[emb_type.lower()]
            cache.create_dataset(emb_type.lower(), data=data, compression="lzf")

        # Build and save neighborhoods
        log.info(f'Building neighborhoods')
        config["emb_types"] = {}
        for emb_type, embedder in embedders.items():
            config["emb_types"][emb_type.lower()] = {}
            config["emb_types"][emb_type.lower()]["metrics"] = []
            for metric in embedder.metrics:
                config["emb_types"][emb_type.lower()]["metrics"].append(metric)
                if embedder.reducer:
                    dims = embedder.reducer.n_components
                else:
                    dims = embedder.feature_length
                config["emb_types"][emb_type.lower()]["dims"] = dims
                ann = AnnoyIndex(dims, metric)
                for i, idx in enumerate(valid_idxs):
                    ann.add_item(i, cache[emb_type.lower()][idx])
                ann.build(n_trees)
                hood_file = os.path.join(model_folder, f"{emb_type.lower()}_{metric}.ann")
                ann.save(hood_file)

        # Align and write metadata
        log.info(f'Aligning metadata')
        meta = []
        for idx in valid_idxs:
            meta.append(X[idx])
        meta_file = os.path.join(model_folder, "metadata.csv")
        csv.writer(open(meta_file, "w")).writerows(meta)

        # Save fitted reducers
        log.info("Saving fitted reducers")
        for emb_type, embedder in embedders.items():
            if embedder.reducer:
                reducer_file = os.path.join(model_folder, f"{emb_type}_reducer.dill")
                with open(reducer_file, "wb") as f:
                    pickle.dump(embedder.reducer, f)

        # Save config
        config_file = os.path.join(model_folder, "config.json")
        with open(config_file, "w") as f:
            json.dump(config, f)

        # Cleanup
        embs.close()
        cache.close()
        os.remove(cache_file)


def make_model(model_folder, 
               data_root, 
               embedders_file="embedders.pytxt", 
               num_workers=32, 
               start=0, 
               end=None, 
               n_trees=100, 
               shuffle=False, 
               max_data=None,
               build=True):
    log.info(f"Creating {model_folder}")
    new_dir(model_folder)

    with open(embedders_file, "r") as f:
        embedders_string = f.read()

    log.info("Creating embedders and writing to file")
    with open(os.path.join(model_folder, "embedders.pytxt"), "w") as f:
        f.write(embedders_string)
    locals = {}
    exec(embedders_string, globals(), locals)
    embedders = locals['embedders']

    X = []
    
    if data_root.endswith(".csv"): # CSV
        with open(data_root, "r") as f:
            meta = csv.reader(f)
            for row in meta:
                X.append(row)
        data_root = None
    else: # Not CSV
        for root, _, files in os.walk(data_root):
            for fname in files:
                X.append([os.path.relpath(os.path.join(root, fname), start=data_root), "", None])
        
    X = arrange_data(X, shuffle, max_data)

    log.info('Training')
    train(
        X=X,
        data_root=data_root,
        model_folder=model_folder,
        embedders=embedders,
        num_workers=num_workers,
        start=start,
        end=end,
        n_trees=n_trees,
        build=build
    )

    log.info('Done')