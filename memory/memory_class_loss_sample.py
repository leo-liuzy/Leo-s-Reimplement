from pytorch_transformers import BertModel
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
import logging
from scipy.special import softmax
logger = logging.getLogger(__name__)


from settings.settings import MODEL_CLASSES
from utils.utils_class import pad_to_max_len


class ClassMemoryLossSample:
    def __init__(self, args):
        self.n_neighbors = args.n_neighbors
        with torch.no_grad():
            logger.info("Initializing memory {} model".format(args.model_name))
            self.model = BertModel.from_pretrained(args.model_name).to(args.devices[0])
            self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        self.max_len = self.model.config.max_position_embeddings
        self.keys, self.input_ids, self.labels = [], [], []
        self.losses = []

        self.tree = NearestNeighbors(n_jobs=args.n_workers)
        self.built_tree = False
        self.devices = args.devices

    def add(self, input_ids: torch.Tensor, masks: torch.Tensor,
            labels: torch.Tensor, losses: torch.Tensor, inds=None):
        """

        :param input_ids: batched
        :param masks: batched
        :param labels: batched
        :param losses: batched
        :return:
        """
        assert len(input_ids) == len(masks) == len(labels) == len(losses)
        if self.built_tree:
            logging.warning("Tree already build! Ignore add.")
            return
        outputs = self.model(input_ids=input_ids, attention_mask=masks)
        if inds is None:
            self.keys.extend(outputs[0][:, 0, :].detach().cpu().tolist())
            for input_id, mask in zip(input_ids.cpu().tolist(), masks.cpu().tolist()):
                min_zero_id = len(mask)
                while mask[min_zero_id-1] == 0:
                    min_zero_id -= 1
                self.input_ids.append(input_id[:min_zero_id])
            self.labels.extend(labels.cpu().tolist())
            self.losses.extend(losses.cpu().tolist())
            del outputs
        else:
            # if the samples are inserted before, just update the loss
            assert len(inds) == len(input_ids) == len(losses)
            for i in range(len(inds)):
                ind = inds[i]
                self.losses[ind] = losses[i].cpu().tolist()

    def sample(self, n_samples):
        if self.built_tree:
            logging.warning("Tree already build! Ignore sample.")
            return
        inds = np.random.choice(len(self.labels), size=n_samples, p=softmax(self.losses))
        input_ids = [self.input_ids[ind] for ind in inds]
        labels = [self.labels[ind] for ind in inds]
        input_ids, masks = pad_to_max_len(input_ids)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids.to(self.devices[0]), masks.to(self.devices[0]), labels.to(self.devices[0]), inds

    def build_tree(self):
        if self.built_tree:
            logging.warning("Tree already build! Ignore build.")
            return
        self.built_tree = True
        self.keys = np.array(self.keys)
        self.tree.fit(self.keys)
        self.input_ids = np.array(self.input_ids)
        self.labels = np.array(self.labels)
        self.losses = np.array(self.losses)

    def query(self, input_ids, masks):
        if not self.built_tree:
            logging.warning("Tree not built! Ignore query.")
            return
        outputs = self.model(input_ids=input_ids, attention_mask=masks)
        queries = outputs[0][:, 0, :].cpu().numpy()
        inds = self.tree.kneighbors(queries, n_neighbors=self.n_neighbors, return_distance=False)
        input_ids, masks = list(zip(*[pad_to_max_len(input_id) for input_id in self.input_ids[inds]]))
        labels = [torch.tensor(label, dtype=torch.long) for label in self.labels[inds]]
        return input_ids, masks, labels
