from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from torch import optim
from torch.utils.data import DataLoader
import argparse
import copy
import logging
import numpy as np
import os
import pickle
import torch
from torch.optim import Adam
import shutil
from ipdb import set_trace as bp
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from settings import parse_test_args, MODEL_CLASSES, init_logging
from utils_class import BertTextClassificationDataset, class_dynamic_collate_fn, prepare_inputs, \
    DynamicBatchSampler, batch_from_numpy_to_tensor, prepare_test_example
from utils_original import TextClassificationDataset


def update_metrics(loss, logits, cur_loss, cur_acc, labels):
    preds = np.argmax(logits, axis=1)
    return cur_loss + loss, cur_acc + np.sum(preds == labels.detach().cpu().numpy())


def local_adapt(args, input_ids, label, tmp_model, q_retrieved_examples, org_params):
    """
    return lists of predictions and classification losses of length (adapt_steps + 1)
    [result(acc/loss) after i adapt step(s) : i=0...n]
    """

    q_retrieved_examples = batch_from_numpy_to_tensor(args, q_retrieved_examples)

    optimizer = Adam(tmp_model.parameters(), lr=args.adapt_lr, eps=args.adam_epsilon)

    def predict(model, input_ids, label):
        with torch.no_grad():
            tmp_model.eval()
            output = model(input_ids=input_ids, labels=label)[:2]
            loss = output[0].item()
            logits = output[1].detach().cpu().numpy()
            torch.cuda.empty_cache()
            return loss, logits

    accs = []
    losses = []

    for step in range(args.adapt_steps + 1):
        # make predictions
        loss, logits = predict(tmp_model, input_ids, label)
        losses.append(loss)
        accs.append(np.argmax(logits, axis=1) == label)
        if step == args.adapt_steps:
            break
        # make 1 local step
        tmp_model.train()
        params = torch.cat([torch.reshape(param, [-1]) for param in tmp_model.parameters()], 0)
        output = tmp_model(input_ids=q_input_ids, attention_mask=q_masks, labels=q_labels)[:2]
        loss = output[0] + args.adapt_lambda * torch.sum((org_params - params) ** 2)
        tmp_model.zero_grad()
        loss.backward()
        optimizer.step()

    assert len(accs) == len(losses) == args.adapt_steps + 1
    return accs, losses


def plot_acc_and_loss(args, accs, losses, file_name):
    assert len(accs) == len(losses) == args.n_test
    mean_losses = np.mean(losses, axis=0)
    mean_acces = np.mean(accs, axis=0)
    local_steps = np.arange(0, len(mean_acces))
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(local_steps, mean_losses, marker='o')
    ax2.plot(local_steps, mean_acces, marker='o')
    ax1.set_ylabel("loss")
    ax2.set_ylabel("acc")
    ax1.get_shared_x_axes().join(ax1, ax2)
    plt.savefig(f"{file_name}_testsize{args.n_test}.png")


def test(dataset_id, args, model, test_dataset):

    # if not args.no_fp16_test:
    #     model = model.half()

    cur_loss, cur_acc = 0, 0
    idxs = np.arange(len(test_dataset))
    if args.random_sample:
        np.random.shuffle(idxs)
    idxs = idxs[:args.n_test]

    if args.adapt_steps >= 1:
        cur_accs = []
        cur_losses = []
        with torch.no_grad():
            org_params = torch.cat([torch.reshape(param, [-1]) for param in model.parameters()], 0)

        q_retrieved_examples = pickle.load(open(os.path.join(args.output_dir,
                                                             f'q_retrieved_examples-{dataset_id}'), 'rb'))

        for i in idxs:
            test_example = prepare_test_example(args, test_dataset[i])
            accs, losses = local_adapt(args, test_example, copy.deepcopy(model), q_retrieved_examples[i], org_params)
            if len(cur_accs) == len(cur_losses) == 0:
                cur_losses = np.array([losses])
                cur_accs = np.array([accs])
            else:
                cur_losses = np.concatenate([cur_losses, [cur_loss]])
                cur_accs = np.concatenate([cur_accs, [cur_acc]])

            if (i+1) % args.logging_steps == 0:
                logging.info(f"Local adapted {i + 1}/{args.n_test} examples , "
                             f"test loss: {cur_losses.mean(axis=0)[args.adapt_steps]:.3f} , "
                             f"test acc: {cur_accs.mean(axis=0)[args.adapt_steps]:.3f}")
        pickle.dump({"accuracy": cur_accs, "loss": cur_losses},
                    open(f"{args.output_dir}/metrics_against_adapt_step_{args.dataset_id}"))
        plot_acc_and_loss(args, accs=cur_accs, losses=cur_losses,
                          file_name=f"{args.output_dir}/metrics_against_adapt_step_plot_{args.dataset_id}.png")

    else:
        test_dataloader = DataLoader(test_dataset, num_workers=args.n_workers, batch_size=args.batch_size)
        tot_n_inputs = 0
        for step, batch in enumerate(test_dataloader):
            n_inputs = batch[0].shape[0]
            inputs = prepare_inputs(args, batch)
            labels = inputs["labels"]
            tot_n_inputs += n_inputs
            with torch.no_grad():
                model.eval()
                outputs = model(**inputs)
                loss = outputs[0].item()
                logits = outputs[1].detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            cur_loss, cur_acc = cur_loss + loss * n_inputs, cur_acc + np.sum(preds == labels.detach().cpu().numpy())
            if (step+1) % args.logging_steps == 0:
                logging.info(f"Tested {tot_n_inputs}/{len(test_dataset)} examples , "
                             f"test loss: {cur_loss/tot_n_inputs:.3f} , "
                             f"test acc: {cur_acc/tot_n_inputs:.3f}")
        assert tot_n_inputs == len(test_dataset)

    logger.info(f"test loss: {cur_loss / args.n_test :.3f} , "
                f"test acc: {cur_acc / args.n_test:.3f}")
    return cur_acc / args.n_test


def main():
    args = parse_test_args()
    train_args = pickle.load(open(os.path.join(args.output_dir, 'train_args'), 'rb'))
    # bp()
    train_args.model_type = "bert-class"
    train_args.__dict__.update(args.__dict__)
    args = train_args
    init_logging(os.path.join(args.output_dir, args.test_log_filename.split(".")[0] + args.model_type + ".txt"))
    logger.info(f"args: {args}")

    config_class, model_class, args.tokenizer_class = MODEL_CLASSES[args.model_type]
    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels,
                                                hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    args.dataset_order = ["dbpedia_csv", "yahoo_answers_csv",  "ag_news_csv",  "amazon_review_full_csv", "yelp_review_full_csv"]
    save_model_path = os.path.join(args.output_dir, f'checkpoint-{len(args.dataset_order)-1}')
    model = model_class.from_pretrained(save_model_path, config=model_config).to(args.device)

    avg_acc = 0
    for dataset_id, dataset_name in enumerate(args.dataset_order):
        logger.info(f"Start testing {dataset_name}...")
        test_dataset = pickle.load(open(os.path.join(args.output_dir, f'test_dataset-{dataset_id}'), 'rb'))
        dataset_acc = test(dataset_id, args, model, test_dataset)
        avg_acc += dataset_acc / len(args.dataset_order)
    logger.info(f"Average acc: {avg_acc:.3f}")


if __name__ == "__main__":
    main()
