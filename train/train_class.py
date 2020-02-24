from torch.utils.data import DataLoader
import logging
import numpy as np
import os
import pickle
import torch
from torch import optim
from math import ceil

logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from memory.memory_class import ClassMemory
from settings.settings import parse_train_args, MODEL_CLASSES, init_logging
from utils.utils_class import TextClassificationDataset, DynamicBatchSampler, dynamic_collate_fn, prepare_inputs


def query_neighbors(task_id, args, memory, test_dataset):
    test_dataloader = DataLoader(test_dataset, num_workers=args.n_workers, collate_fn=dynamic_collate_fn,
                                 batch_sampler=DynamicBatchSampler(test_dataset, args.batch_size, mode="seq"))

    q_input_ids, q_masks, q_labels = [], [], []
    for step, batch in enumerate(test_dataloader):
        n_inputs, input_ids, masks, labels = prepare_inputs(args, batch)
        with torch.no_grad():
            cur_q_input_ids, cur_q_masks, cur_q_labels = memory.query(input_ids, masks)
        q_input_ids.extend(cur_q_input_ids)
        q_masks.extend(cur_q_masks)
        q_labels.extend(cur_q_labels)
        if (step + 1) % args.logging_steps == 0:
            logging.info(f"Queried {len(q_masks)} examples")
    pickle.dump(q_input_ids, open(os.path.join(args.output_dir, f'q_input_ids-{task_id}'), 'wb'))
    pickle.dump(q_masks, open(os.path.join(args.output_dir, f'q_masks-{task_id}'), 'wb'))
    pickle.dump(q_labels, open(os.path.join(args.output_dir, f'q_labels-{task_id}'), 'wb'))


def train_task(args, model, memory, optimizer, train_dataset, valid_dataset=None):
    train_dataloader = DataLoader(train_dataset, num_workers=args.n_workers, collate_fn=dynamic_collate_fn,
                                  batch_sampler=DynamicBatchSampler(train_dataset, args.batch_size,
                                                                    mode=args.sampler_choice))
    # if valid_dataset:
    #     valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size * 6,
    #                                   num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
    # model.zero_grad()
    tot_epoch_loss, tot_n_inputs = 0, 0

    for step, batch in enumerate(train_dataloader):
        model.train()
        n_inputs, input_ids, masks, labels = prepare_inputs(args, batch)
        if np.random.rand() < args.write_ratio:
            memory.add(input_ids, masks, labels)
        loss = model(input_ids=input_ids, attention_mask=masks, labels=labels)[0]

        model.zero_grad()
        loss.backward()
        optimizer.step()
        tot_n_inputs += n_inputs
        tot_epoch_loss += loss.item() * n_inputs

        del loss
        if (step + 1) % args.logging_steps == 0:
            logger.info(f"progress: {tot_n_inputs / args.n_train:.2f} , step: {step + 1} , "
                        f"lr: {optimizer.param_groups[0]['lr']:.2E} , "
                        f"avg batch size: {tot_n_inputs // (step + 1):.1f} , "
                        f"avg loss: {tot_epoch_loss / tot_n_inputs:.3f}")

        if args.replay_interval >= 1 and (step + 1) % (args.replay_interval // args.batch_size) == 0:
            torch.cuda.empty_cache()
            input_ids, masks, labels = memory.sample(args.replay_sample)
            num_batch = ceil(args.replay_sample / args.batch_size)
            for i in range(num_batch):
                l = i * args.batch_size
                u = (i + 1) * args.batch_size
                input_id = input_ids[l:u]
                mask = masks[l:u]
                label = labels[l:u]
                loss = model(input_ids=input_id, attention_mask=mask, labels=label)[0]

                model.zero_grad()
                loss.backward()
                optimizer.step()

        torch.cuda.empty_cache()

    # del train_dataset
    logger.info(f"Finsih training, avg loss: {tot_epoch_loss / tot_n_inputs :.3f}")
    assert tot_n_inputs == len(train_dataset) == args.n_train


def main():
    args = parse_train_args()
    pickle.dump(args, open(os.path.join(args.output_dir, 'train_args'), 'wb'))
    init_logging(os.path.join(args.output_dir, 'log_train.txt'))
    logger.info("args: " + str(args))

    logger.info(f"Initializing main {args.model_name} model")
    config_class, model_class, args.tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = args.tokenizer_class.from_pretrained(args.model_name)

    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels)
    config_save_path = os.path.join(args.output_dir, 'config')
    model_config.to_json_file(config_save_path)
    model = model_class.from_pretrained(args.model_name, config=model_config)
    model = torch.nn.DataParallel(model, args.device_ids)
    memory = ClassMemory(args)

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

        logger.info(f"Start training {task}...")
        train_task(args, model, memory, optimizer, train_dataset, valid_dataset)
        del train_dataset
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
        query_neighbors(task_id, args, memory, test_dataset)


if __name__ == "__main__":
    main()
