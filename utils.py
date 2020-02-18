from multiprocessing import Pool
from torch.utils.data import Dataset, Sampler
from transformers import BertTokenizer
import csv
import numpy as np
import os
import random
import torch
from argparse import Namespace
from utils_squad import read_squad_examples, convert_examples_to_features

from settings import label_offsets, TASKS
from scipy.stats import describe
count_token = lambda x: len(x.split(" "))


def pad_to_max_len(input_ids, masks=None):
    max_len = max(len(input_id) for input_id in input_ids)
    masks = torch.tensor([[1]*len(input_id)+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    input_ids = torch.tensor([input_id+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    return input_ids, masks


def class_dynamic_collate_fn(batch):
    labels, input_ids = list(zip(*batch))
    labels = torch.tensor([b[0] for b in batch], dtype=torch.long)
    input_ids, masks = pad_to_max_len(input_ids)
    return input_ids, masks, labels


def read_from_csv(fname, delimiter, quotechar):
    data = []
    with open(fname, 'r', encoding='utf-8') as f:
        for row in csv.reader(f, delimiter=delimiter, quotechar=quotechar):
            data.append(row)
    return data


def prepare_inputs(args, batch):
    assert args.task in TASKS

    if args.task == "qa":
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'start_positions': batch[3].to(args.device),
                  'end_positions': batch[4].to(args.device)}
    else:
        inputs = {"input_ids": batch[0].to(args.device),
                  "attention_mask": batch[1].to(args.device),
                  "labels": batch[2].to(args.device)}
    return inputs


def load_and_cache_examples(input_file, args, tokenizer, evaluate=False, output_examples=False, logger=None):
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'test' if evaluate else 'train',
        args.model_type,
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        if logger is not None:
            logger.info(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        if logger is not None:
            logger.info(f"Creating features from dataset file at {input_file}", )
        examples = read_squad_examples(input_file=input_file,
                                       is_training=not evaluate,
                                       version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = list(zip(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask))
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = list(zip(all_input_ids, all_input_mask, all_segment_ids,
                           all_start_positions, all_end_positions,
                           all_cls_index, all_p_mask))
        if logger is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if output_examples:
        return dataset, examples, features
    return dataset


class ContinualLearningDataset(Dataset):
    def __init__(self, dataset_dir: str, mode: str, args: Namespace, tokenizer: BertTokenizer, logger):
        self.dataset_name = dataset_dir
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = 250  # 512 for bert tokenizer
        self.n_test = args.n_test
        self.n_train = args.n_train
        self.valid_ratio = args.valid_ratio
        self.file_type = ".json" if args.task in ["qa"] else ".csv"
        self.data = []
        self.logger = logger

        if self.mode == "test":
            self.fname = os.path.join(dataset_dir, "test" + self.file_type)
        elif self.mode in ["train", "valid"]:
            self.fname = os.path.join(dataset_dir, "train" + self.file_type)

        self.split_start = 0
        if mode == "test":
            self.split_end = self.n_test
        elif mode == "valid":
            self.split_end = int(self.n_train * self.valid_ratio)
        elif mode == "train":
            self.split_start = int(self.n_train * self.valid_ratio)
            self.split_end = self.n_train
        else:
            raise Exception("Mode Error")

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class BertQuestionAnsweringDataset(ContinualLearningDataset):
    def __init__(self, dataset_dir: str, mode: str, args: Namespace, tokenizer: BertTokenizer, logger):
        super(BertQuestionAnsweringDataset, self).__init__(dataset_dir, mode, args, tokenizer, logger)

        self.data = load_and_cache_examples(self.fname, args, tokenizer,
                                            evaluate=mode == "test", logger=logger)

        random.shuffle(self.data)
        self.data = self.data[self.split_start:self.split_end]

        with Pool(args.n_workers) as pool:
            self.data = pool.map(lambda x: x, self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BertTextClassificationDataset(ContinualLearningDataset):
    def __init__(self, dataset_dir: str, mode: str, args: Namespace, tokenizer: BertTokenizer, logger):

        super(BertTextClassificationDataset, self).__init__(dataset_dir, mode, args, tokenizer, logger)
        # self.max_len = 128
        self.label_offset = label_offsets[dataset_dir.split('/')[-1]]

        self.data = read_from_csv(self.fname, delimiter=',', quotechar='"')
        random.shuffle(self.data)
        self.data = self.data[:100]  # self.split_start:self.split_end]
        logger.info(f"{self.fname} stats: {describe(list(map(count_token, self.data)))}")
        # with Pool(args.n_workers) as pool:
        #     self.data = pool.map(self.map_csv, self.data)
        self.data = list(map(self.map_csv, self.data))

    def _add_spl_ids_and_pad(self, input_ids,):
        if len(input_ids) > self.max_len-2:
            input_ids = [self.tokenizer.cls_token_id] + \
                    input_ids[:self.max_len-2] + [self.tokenizer.sep_token_id]
            return input_ids
        
        output = [self.tokenizer.cls_token_id]
        output.extend(input_ids)
        output.append(self.tokenizer.sep_token_id)
        padding = [0]*(self.max_len-len(output))
        output.extend(padding)
        return output

    def __getitem__(self, idx):
        return self.data[idx]

    def map_csv(self, row):
        context = '[CLS]' + ' '.join(row[1:])[:self.max_len-2] + '[SEP]'
        return int(row[0]) + self.label_offset, self.tokenizer.encode(context)


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, mode="seq"):
        super(DynamicBatchSampler, self).__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_samples = len(dataset)
        self.idxs = None
        if mode == "seq":
            self.idxs = np.arange(len(dataset))
        else:
            assert mode == "random"
            self.idxs = np.random.randint(self.n_samples, size=(self.n_samples,), dtype=np.int32)

    def __iter__(self):
        max_len = 0
        batch = []
        for idx in self.idxs:
            # max_len = max(max_len, len(self.dataset[idx][1]))
            if len(batch) == self.batch_size:
                print(batch)
                yield batch
                max_len = 0
                batch = []
            # max_len = max(max_len, len(self.dataset[idx][1]))
            batch.append(idx)
        if len(batch) > 0:
            yield batch


class TextClassificationDataset(Dataset):
    def __init__(self, task, mode, args, tokenizer):
        self.task = task
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = tokenizer.max_len
        self.n_test = args.n_test
        self.n_train = args.n_train
        self.valid_ratio = args.valid_ratio

        self.data = []
        self.label_offset = label_offsets[task.split('/')[-1]]
        if self.mode == "test":
            self.fname = os.path.join(task, "test.csv")
        elif self.mode in ["train", "valid"]:
            self.fname = os.path.join(task, "train.csv")

        with open(self.fname, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                self.data.append(row)

        random.shuffle(self.data)

        if mode == "test":
            self.data = self.data[:self.n_test]
        elif mode == "valid":
            self.data = self.data[:int(self.n_train * self.valid_ratio)]
        elif mode == "train":
            self.data = self.data[int(self.n_train * self.valid_ratio): self.n_train]

        with Pool(args.n_workers) as pool:
            self.data = pool.map(self.map_csv, self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map_csv(self, row):
        context = '[CLS]' + ' '.join(row[1:])[:self.max_len-2] + '[SEP]'
        return (int(row[0]) + self.label_offset, self.tokenizer.encode(context))

