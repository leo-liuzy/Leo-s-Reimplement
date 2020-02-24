from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, SequentialSampler, Dataset
import logging
import numpy as np
import os
import pickle
import torch
from torch import optim
from ipdb import set_trace as bp
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from memory import Memory, QAMemory, ClassificationMemory
from settings import parse_train_args, MODEL_CLASSES, init_logging
from utils import TextClassificationDataset, DynamicBatchSampler
from utils import class_dynamic_collate_fn, prepare_inputs, batch_from_tensor_to_numpy


def query_neighbors(args, task_id: int, memory: Memory, test_dataset: Dataset):
    test_dataloader = DataLoader(test_dataset, num_workers=args.n_workers, batch_size=args.batch_size)

    q_retrieved_examples = []
    for step, batch in enumerate(test_dataloader):
        batch = prepare_inputs(args, batch)
        with torch.no_grad():
            retrieved_examples = memory.query(**batch)
            retrieved_examples = batch_from_tensor_to_numpy(retrieved_examples)
        q_retrieved_examples.append(retrieved_examples)
        if (step+1) % args.logging_steps == 0:
            logging.info(f"Queried {len(q_retrieved_examples)} examples")
    pickle.dump(q_retrieved_examples,
                open(os.path.join(args.output_dir, f'q_retrieved_examples-{task_id}'), 'wb'))


def train_task(args, model, memory, optimizer, train_dataset, valid_dataset):

    train_dataloader = DataLoader(train_dataset, num_workers=args.n_workers, collate_fn=dynamic_collate_fn,
            batch_sampler=DynamicBatchSampler(train_dataset, args.batch_size, mode=args.sampler_choice))
    # if valid_dataset:
    #     valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size * 6,
    #                                   num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
    tot_epoch_loss, tot_n_inputs = 0, 0

    def update_parameters(loss):
        model.zero_grad()
        loss.backward()
        optimizer.step()

    for step, batch in enumerate(train_dataloader):
        model.train()
        n_inputs = len(batch)
        inputs = prepare_inputs(args, batch)
        if np.random.rand() < args.write_ratio:
            memory.add(**inputs)
        loss = model(**inputs)[0]
        update_parameters(loss)
        tot_n_inputs += n_inputs
        tot_epoch_loss += loss.item() * n_inputs

        if (step+1) % args.logging_steps == 0:
            logger.info(f"progress: {tot_n_inputs/args.n_train :.2f} , "
                        f"step: {step+1} , "
                        f"lr: {optimizer.param_groups[0]['lr'] :.2E} , "
                        f"avg batch size: {tot_n_inputs//(step+1) :.1f} , "
                        f"avg loss: {tot_epoch_loss/tot_n_inputs:.3f}")

        if args.replay_interval >= 1 and (step+1) % args.replay_interval == 0:
            torch.cuda.empty_cache()
            sampled_query = memory.sample(tot_n_inputs // (step + 1))
            loss = model(**sampled_query)[0]
            update_parameters(loss)

    logger.info(f"Finsih training, avg loss: {tot_epoch_loss/tot_n_inputs :.3f}")
    assert tot_n_inputs == len(train_dataset) == args.n_train


def main():
    args = parse_train_args()
    pickle.dump(args, open(os.path.join(args.output_dir, 'train_args'), 'wb'))
    init_logging(os.path.join(args.output_dir, 'log_train.txt'))
    logger.info(f"args: {args}")

    logger.info(f"Initializing main {args.model_name} model")
    config_class, model_class, args.tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = args.tokenizer_class.from_pretrained(args.model_name)

    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels)
    config_save_path = os.path.join(args.output_dir, 'config')
    model_config.to_json_file(config_save_path)
    model = model_class.from_pretrained(args.model_name, config=model_config).cuda()
    memory = QAMemory(args) if args.task == "qa" else ClassificationMemory(args)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    for task_id, task in enumerate(args.tasks):
        logger.info(f"Start parsing {task} train data...")
        train_dataset = TextClassificationDataset(task, "train", args, tokenizer)

        logger.info(f"Max length: {train_dataset.max_len}...")
        if args.valid_ratio > 0:
            logger.info(f"Start parsing {task} valid data...")
            valid_dataset = TextClassificationDataset(task, "valid", args, tokenizer)
        else:
            valid_dataset = None

        logger.info("Start training {}...".format(task))
        train_task(args, model, memory, optimizer, train_dataset, valid_dataset)
        model_save_path = os.path.join(args.output_dir, f'checkpoint-{task_id}')
        torch.save(model.state_dict(), model_save_path)
        pickle.dump(memory, open(os.path.join(args.output_dir, f'memory-{task_id}'), 'wb'))

    del model
    memory.build_tree()

    for task_id, task in enumerate(args.tasks):
        logger.info(f"Start parsing {task} test data...")
        test_dataset = TextClassificationDataset(task, "test", args, tokenizer)
        pickle.dump(test_dataset, open(os.path.join(args.output_dir, f'test_dataset-{task_id}'), 'wb'))
        logger.info(f"Start querying {task}...")
        query_neighbors(args, task_id, memory, test_dataset)


if __name__ == "__main__":
    main()
