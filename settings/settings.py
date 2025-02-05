from transformers import BertForSequenceClassification, BertForQuestionAnswering,\
    BertTokenizer, BertConfig
import GPUtil
import argparse
import datetime
import logging
import os
import shutil
import psutil
import torch

from ipdb import set_trace as bp

MAX_TRAIN_SIZE = 115000
MAX_TEST_SIZE = 7600
TASKS = ["text_classification", "qa"]
SAMPLERS = ["seq", "random"]
MAX_SEQ_LEN=128
SEED = 42
# MEMORY_SAMPLE = ['random', 'loss']
MODEL_CLASSES = {
    'bert-class': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'bert-qa': (BertConfig, BertForQuestionAnswering, BertTokenizer)
    # 'xlnet': (XLNetForSequenceClassification, XLNetTokenizer),
    # 'xlm': (XLMForSequenceClassification, XLMTokenizer),
}
label_offsets = {
    'ag_news_csv': -1,  # 1-4   # in our 33 labels: 0-3
    'amazon_review_full_csv': 3,  # 1-5    in our 33 labels: 4-8
    'yelp_review_full_csv': 3,  # 1-5   in our 33 labels: 4-8
    'dbpedia_csv': 8,  # 1-14  # in our 33 labels: 9-22
    'yahoo_answers_csv': 22  # 1-10  # in our 33 labels: 23-32
}


def set_device(args):
    if args.num_gpu > 0 and torch.cuda.is_available():
        args.device_ids = GPUtil.getFirstAvailable(maxLoad=0.05, maxMemory=0.05)[:args.num_gpu]
        args.devices = [torch.device(f"cuda:{device_id}") for device_id in args.device_ids]
    else:
        args.devices = [torch.device("cpu")]
        args.device_ids = [-1]


def seed_randomness(args):
    import numpy as np
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)


def parse_train_args():
    parser = argparse.ArgumentParser("Train Lifelong Language Learning")

    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sampler_choice", type=str, default="random", choices=SAMPLERS)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_gpu", nargs='+', type=int, default=-1)
    parser.add_argument("--model_type", type=str, default="bert-class",
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--n_labels", type=int, default=33)
    parser.add_argument("--n_neighbors", type=int, default=32)
    parser.add_argument("--n_test", type=int, default=MAX_TEST_SIZE)
    parser.add_argument("--n_train", type=int, default=MAX_TRAIN_SIZE)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--output_dir", type=str, default="output0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replay_sample", type=int, default=100)
    parser.add_argument("--replay_interval", type=int, default=10000)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--tasks", nargs='+', default=["ag_news_csv"])
    # parser.add_argument("--sample_strategy", type=str, default="random", choices=MEMORY_SAMPLE)
    parser.add_argument("--write_ratio", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument("--task", type=str, default=TASKS[0],
                        choices=TASKS)
    parser.add_argument("--train_log_filename", type=str, default="log_train.txt")

    args = parser.parse_args()

    if args.debug:
        args.n_train = 500
        args.logging_steps = 1
        args.n_test = 100
        args.output_dir = "output_debug"
        args.overwrite = True

    set_device(args)

    if args.device_id == -1:
        memory_size = psutil.virtual_memory().available // 2 ** 20  # turn unit into MB
    else:
        memory_size = GPUtil.getGPUs()[args.device_id].memoryTotal

    if args.batch_size <= 0:
        args.batch_size = int(memory_size * 0.38)

    if os.path.exists(args.output_dir):
        if args.overwrite:
            choice = 'y'
        else:
            choice = input(f"Output directory ({args.output_dir}) exists! Remove? ")
        if choice.lower()[0] == 'y':
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            raise ValueError("Output directory exists!")
    else:
        os.makedirs(args.output_dir)

    tmp_train_log_file = os.path.join(args.output_dir, args.train_log_filename)
    if os.path.exists(tmp_train_log_file):
        choice = input(f"Train log ({args.train_log_filename}) exists! Remove? ")
        if choice.lower()[0] == 'y':
            os.remove(tmp_train_log_file)
        else:
            raise ValueError("Train log exists!")

    return args


def parse_test_args():
    parser = argparse.ArgumentParser("Test Lifelong Language Learning")

    parser.add_argument("--adapt_lambda", type=float, default=1e-3)
    parser.add_argument("--adapt_lr", type=float, default=5e-3)
    parser.add_argument("--adapt_steps", type=int, default=30)
    parser.add_argument("--fp16_test", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output0")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--test_log_filename", type=str, default="log_test.txt")
    parser.add_argument('--seed', type=int, default=SEED)

    args = parser.parse_args()
    args.test_log_filename = aug_log_file(args, args.test_log_filename)
    seed_randomness(args)
    set_device(args)
    assert args.n_test <= MAX_TEST_SIZE

    tmp_test_log_file = os.path.join(args.output_dir, args.test_log_filename)
    if os.path.exists(tmp_test_log_file):
        choice = input(f"Test log ({args.test_log_filename}) exists! Remove? ")
        if choice.lower()[0] == 'y':
            os.remove(tmp_test_log_file)
        else:
            raise ValueError("Test log exists!")

    return args


def aug_log_file(args, filename):
    prefix, suffix = filename.split(".")
    name = f"{prefix}_adapstep{args.adapt_steps}_ntest{args.n_test}_loclr{args.adapt_lr}"
    return name + "." + suffix


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated

        delta = record.relativeCreated/1000 - last/1000
        record.relative = f"{delta:.3f}"
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
        self.last = record.relativeCreated
        return True


def init_logging(filename):
    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format, filename=filename, filemode='a', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

