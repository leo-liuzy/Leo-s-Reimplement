from pytorch_transformers import BertModel
from sklearn.neighbors import NearestNeighbors
import torch
from typing import *
import numpy as np
import pickle
import os
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

from settings import MODEL_CLASSES
from utils import pad_to_max_len


class Memory(torch.nn.Module):
    def __init__(self, args):
        super(Memory, self).__init__()
        self.n_neighbors = args.n_neighbors
        with torch.no_grad():
            logger.info("Initializing memory {} model".format(args.model_name))
            self.model = BertModel.from_pretrained(args.model_name).to(args.devices[0])
            self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        self.tree = NearestNeighbors(n_jobs=args.n_workers)
        self.built_tree = None
        self.devices = args.devices

    def add(self, input_ids, masks, **others):
        if self.built_tree:
            logging.warning("Tree already build! Ignore add.")
            return
        outputs = self.model(input_ids=input_ids, attention_mask=masks)
        self.keys.extend(outputs[0][:, 0, :].detach().cpu().tolist())
        for input_id, mask in zip(input_ids.cpu().tolist(), masks.cpu().tolist()):
            min_zero_id = len(mask)
            while mask[min_zero_id-1] == 0:
                min_zero_id -= 1
            self.input_ids.append(input_id[:min_zero_id])
        del outputs

    def sample(self, n_samples):
        if self.built_tree:
            logging.warning("Tree already build! Ignore sample.")
            return
        inds = np.random.randint(len(self.input_ids), size=n_samples)
        input_ids = [self.input_ids[ind] for ind in inds]
        input_ids, masks = pad_to_max_len(input_ids)
        return input_ids, masks, inds

    def build_tree(self):
        if self.built_tree:
            logging.warning("Tree already build! Ignore build.")
            return
        self.built_tree = True
        self.keys = np.array(self.keys)
        self.tree.fit(self.keys)
        self.input_ids = np.array(self.input_ids)

    def query(self, **inputs):
        if not self.built_tree:
            logging.warning("Tree not built! Ignore query.")
            return
        outputs = self.model(**inputs)
        queries = outputs[0][:, 0, :].cpu().numpy()
        inds = self.tree.kneighbors(queries, n_neighbors=self.n_neighbors, return_distance=False)
        input_ids, masks = list(zip(*[pad_to_max_len(input_id) [0]for input_id in self.input_ids[inds]]))
        return input_ids, masks, inds


class ClassificationMemory(Memory):
    def __init__(self, args):
        super(ClassificationMemory, self).__init__(args)
        self.hidden_size = self.model.config.hidden_size
        self.max_len = self.model.config.max_position_embeddings
        self.keys, self.input_ids, self.labels = [], [], []

    def add(self, input_ids, masks, **others):
        super().add(input_ids, masks)
        self.labels.extend(others["labels"].cpu().tolist())

    def sample(self, n_samples) -> Dict[str, torch.Tensor]:
        input_ids, masks, inds = super().sample(n_samples)
        labels = torch.tensor([self.labels[ind] for ind in inds], dtype=torch.long)
        retval = {
            "input_ids": input_ids.to(self.devices[0]),
            "attention_mask": masks.to(self.devices[0]),
            "labels": labels.to(self.devices[0])
        }
        return retval

    def build_tree(self):
        super().build_tree()
        self.labels = np.array(self.labels)

    def query(self, **input) -> Dict[str, torch.Tensor]:
        input_ids, masks, inds = super().query(**input)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.long)
        labels = torch.tensor([torch.tensor(label, dtype=torch.long)
                               for label in self.labels[inds]])
        retval = {
            "input_ids": input_ids.to(self.devices[0]),
            "attention_mask": masks.to(self.devices[0]),
            "labels": labels.to(self.devices[0])
        }
        return retval


class QAMemory(Memory):
    def __init__(self, args):
        super(QAMemory, self).__init__(args)
        self.max_len = self.model.config.max_position_embeddings
        self.keys, self.input_ids, self.start_positions, self.end_positions = [], [], [], []

    def add(self, input_ids, masks, **others):
        super().add(input_ids, masks)
        self.start_positions.extend(others["start_positions"].cpu().tolist())
        self.end_positions.extend(others["end_positions"].cpu().tolist())

    def sample(self, n_samples) -> Dict[str, torch.Tensor]:
        input_ids, masks, inds = super().sample(n_samples)
        start_positions = torch.tensor([self.start_positions[ind] for ind in inds], dtype=torch.long)
        end_positions = torch.tensor([self.end_positions[ind] for ind in inds], dtype=torch.long)
        retval = {
            "input_ids": input_ids.to(self.devices[0]),
            "attention_mask": masks.to(self.devices[0]),
            "start_positions": start_positions.to(self.devices[0]),
            "end_positions": end_positions.to(self.devices[0])
        }
        return retval

    def build_tree(self):
        super().build_tree()
        self.start_positions = np.array(self.start_positions)
        self.end_positions = np.array(self.end_positions)

    def query(self, **input) -> Dict[str, torch.Tensor]:
        input_ids, masks, inds = super().query(**input)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.long)
        start_positions = [torch.tensor(start_position, dtype=torch.long)
                           for start_position in self.start_positions[inds]]
        end_positions = [torch.tensor(end_position, dtype=torch.long)
                         for end_position in self.end_positions[inds]]
        start_positions = torch.tensor(start_positions)
        end_positions = torch.tensor(end_positions)
        retval = {
            "input_ids": input_ids.to(self.devices[0]),
            "attention_mask": masks.to(self.devices[0]),
            "start_positions": start_positions.to(self.devices[0]),
            "end_positions": end_positions.to(self.devices[0])
        }
        return retval
