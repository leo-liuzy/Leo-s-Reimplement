from transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, SequentialSampler
import logging
import numpy as np
import os
from torch import optim
optim.Adam
import pickle
import torch
from ipdb import set_trace as bp
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)


from memory import ClassificationMemory, QAMemory
from settings import parse_train_args, MODEL_CLASSES, init_logging
from utils import BertTextClassificationDataset, DynamicBatchSampler
from utils import class_dynamic_collate_fn, prepare_inputs


def query_neighbors(dataset_id, args, memory, test_dataset):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, num_workers=args.n_workers,
                                 batch_size=args.batch_size, sampler=test_sampler)
    memory.eval()
    q_input_ids, q_masks, q_labels = [], [], []
    for step, batch in enumerate(test_dataloader):
        input_ids, masks, labels = prepare_inputs(args, batch)
        with torch.no_grad():
            cur_q_input_ids, cur_q_masks, cur_q_labels = memory.query(input_ids, masks)
        q_input_ids.extend(cur_q_input_ids)
        q_masks.extend(cur_q_masks)
        q_labels.extend(cur_q_labels)
        if (step+1) % args.logging_steps == 0:
            logging.info("Queried {} examples".format(len(q_masks)))

    pickle.dump(q_input_ids, open(os.path.join(args.output_dir, 'q_input_ids-{}'.format(dataset_id)), 'wb'))
    pickle.dump(q_masks, open(os.path.join(args.output_dir, 'q_masks-{}'.format(dataset_id)), 'wb'))
    pickle.dump(q_labels, open(os.path.join(args.output_dir, 'q_labels-{}'.format(dataset_id)), 'wb'))


def train(args, model, memory, train_dataset, valid_dataset):

    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=args.n_workers,collate_fn=class_dynamic_collate_fn,
                                 batch_sampler=DynamicBatchSampler(train_dataset, args.batch_size))
    # if valid_dataset:
    #     valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size * 6,
    #                                   num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=len(train_dataset)//10)

    tot_epoch_loss, tot_n_inputs = 0, 0

    def update_parameters(loss):
        model.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        # scheduler.step()

    model.train()
    for step, batch in enumerate(train_dataloader):
        n_inputs = batch[0].shape[0]
        inputs = prepare_inputs(args, batch)  # prepare task-specific input
        memory.add(**inputs)
        loss = model(**inputs)[0]
        update_parameters(loss)
        tot_n_inputs += n_inputs
        tot_epoch_loss += loss.item() * n_inputs

        if (step+1) % args.logging_steps == 0:
            logger.info("progress: {:.2f} , "
                        "step: {} , "
                        "lr: {:.2E} , "
                        "avg batch size: {:.1f} , "
                        "avg loss: {:.3f}".format(tot_n_inputs/args.n_train,
                                                  step+1, scheduler.get_lr()[0],
                                                  tot_n_inputs//(step+1),
                                                  tot_epoch_loss/tot_n_inputs))

        if args.replay_interval >= 1 and (step+1) % args.replay_interval == 0:
            torch.cuda.empty_cache()
            del loss, input_ids, masks, labels
            input_ids, masks, labels = memory.sample(tot_n_inputs // (step + 1))
            loss = model(input_ids=input_ids, attention_mask=masks, labels=labels)[0]
            update_parameters(loss)

    logger.info("Finsih training, avg loss: {:.3f}".format(tot_epoch_loss/tot_n_inputs))
    del optimizer, optimizer_grouped_parameters
    assert tot_n_inputs == len(train_dataset) == args.n_train


def main():
    args = parse_train_args()
    pickle.dump(args, open(os.path.join(args.output_dir, 'train_args'), 'wb'))
    init_logging(os.path.join(args.output_dir, f"{args.train_log_filename.split('.')[0]}_{args.model_type}.txt"))
    logger.info("args: " + str(args))

    logger.info(f"Initializing main {args.model_name} model")
    config_class, model_class, args.tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = args.tokenizer_class.from_pretrained(args.model_name)
    if args.task == "text_classification":
        model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels)
    else:
        model_config = config_class.from_pretrained(args.model_name)
    config_save_path = os.path.join(args.output_dir, 'config' + f"{args.model_name}")
    model_config.to_json_file(config_save_path)
    model = model_class.from_pretrained(args.model_name, config=model_config).to(args.device)
    memory = QAMemory(args) if args.task == "qa" else ClassificationMemory(args)

    for dataset_id, dataset_name in enumerate(args.dataset_order):

        logger.info(f"Start parsing {dataset_name} train data...")
        dataset_dir = f"{args.data_dir}/{dataset_name}"
        train_dataset = BertTextClassificationDataset(dataset_dir, "train", args, tokenizer, logger)

        if args.valid_ratio > 0:
            logger.info(f"Start parsing {dataset_name} valid data...")
            valid_dataset = BertTextClassificationDataset(dataset_dir, "valid", args, tokenizer, logger)
        else:
            valid_dataset = None

        logger.info(f"Start training {dataset_name}...")
        train(args, model, memory, train_dataset, valid_dataset)
        model_save_path = os.path.join(args.output_dir, f'checkpoint-{dataset_id}')
        torch.save(model.state_dict(), model_save_path)
        pickle.dump(memory, open(os.path.join(args.output_dir, f'memory-{dataset_id}'), 'wb'))

    del model
    memory.build_tree()

    for dataset_id, dataset_name in enumerate(args.dataset_order):
        logger.info(f"Start parsing {dataset_name} test data...")
        test_dataset = BertTextClassificationDataset(dataset_name, "test", args, tokenizer, logger)
        pickle.dump(test_dataset, open(os.path.join(args.output_dir, f'test_dataset-{dataset_id}')), 'wb')
        logger.info(f"Start querying {dataset_name}...")
        query_neighbors(dataset_id, args, memory, test_dataset)


if __name__ == "__main__":
    main()
