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
import shutil
from ipdb import set_trace as bp
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from settings import parse_test_args, MODEL_CLASSES, init_logging
from utils import BertTextClassificationDataset, class_dynamic_collate_fn, prepare_inputs, DynamicBatchSampler
from utils_original import TextClassificationDataset


def update_metrics(loss, logits, cur_loss, cur_acc, labels):
    preds = np.argmax(logits, axis=1)
    return cur_loss + loss, cur_acc + np.sum(preds == labels.detach().cpu().numpy())


def local_adapt(input_ids, label, tmp_model, q_input_ids, q_masks, q_labels, args, org_params):
    """
    return lists of predictions and classification losses of length (adapt_steps + 1)
    [result(acc/loss) after i adapt step(s) : i=0...n]
    """

    q_input_ids = q_input_ids.detach().to(args.device)
    q_masks = q_masks.detach().to(args.device)
    q_labels = q_labels.detach().to(args.device)

    optimizer = AdamW(tmp_model.parameters(), lr=args.adapt_lr, eps=args.adam_epsilon)

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
    mean_acces = np.mean()
    local_steps = np.arange(0, len(mean_acces))
    import matplotlib.pyplot as plt
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

        q_input_ids = pickle.load(open(os.path.join(args.output_dir, 'q_input_ids-{}'.format(dataset_id)), 'rb'))
        q_masks = pickle.load(open(os.path.join(args.output_dir, 'q_masks-{}'.format(dataset_id)), 'rb'))
        q_labels = pickle.load(open(os.path.join(args.output_dir, 'q_labels-{}'.format(dataset_id)), 'rb'))

        for i in idxs:
            label, input_id = test_dataset[i]
            # input_id, _, labels = test_dataset[i]
            input_id = torch.tensor(input_id).unsqueeze(0).to(args.device)
            label = torch.tensor(label).unsqueeze(0).to(args.device)
            accs, losses = local_adapt(input_id, label, copy.deepcopy(model), q_input_ids[i], q_masks[i], q_labels[i], args, org_params)
            if len(cur_accs) == len(cur_losses) == 0:
                cur_losses = np.array([losses])
                cur_accs = np.array([accs])
            else:
                cur_losses = np.concatenate([cur_losses, [cur_loss]])
                cur_accs = np.concatenate([cur_accs, [cur_acc]])

            if (i+1) % args.logging_steps == 0:
                logging.info("Local adapted {}/{} examples, test loss: {:.3f} , test acc: {:.3f}".format(
                    i+1, args.n_test,
                    cur_losses.mean(axis=0)[args.adapt_steps-1],
                    cur_accs.mean(axis=0)[args.adapt_steps-1]))
        pickle.dump({"accuracy": cur_accs, "loss": cur_losses},
                    open(f"{args.output_dir}/metrics_against_adapt_step_{args.dataset_id}"))
        plot_acc_and_loss(accs=cur_accs, losses=cur_losses,
                          file_name=f"{args.output_dir}/metrics_against_adapt_step_plot_{args.dataset_id}.png")

    else:
        test_dataloader = DataLoader(test_dataset, num_workers=args.n_workers, collate_fn=class_dynamic_collate_fn,
                                     batch_sampler=DynamicBatchSampler(test_dataset, args.batch_size * 4))
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
                logging.info("Tested {}/{} examples , test loss: {:.3f} , test acc: {:.3f}".format(
                    tot_n_inputs, len(test_dataset), cur_loss/tot_n_inputs, cur_acc/tot_n_inputs))
        assert tot_n_inputs == len(test_dataset)

    logger.info("test loss: {:.3f} , test acc: {:.3f}".format(
        cur_loss / args.n_test, cur_acc / args.n_test))
    return cur_acc / args.n_test


def main():
    args = parse_test_args()
    train_args = pickle.load(open(os.path.join(args.output_dir, 'train_args'), 'rb'))
    # bp()
    train_args.model_type = "bert-class"
    # assert train_args.output_dir == args.output_dir
    train_args.__dict__.update(args.__dict__)
    args = train_args
    init_logging(os.path.join(args.output_dir, args.test_log_filename.split(".")[0] + args.model_type + ".txt"))
    logger.info("args: " + str(args))

    config_class, model_class, args.tokenizer_class = MODEL_CLASSES[args.model_type]
    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels,
                                                hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    args.dataset_order = ["dbpedia_csv", "yahoo_answers_csv",  "ag_news_csv",  "amazon_review_full_csv", "yelp_review_full_csv"]
    save_model_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(len(args.dataset_order)-1))
    model = model_class.from_pretrained(save_model_path, config=model_config).to(args.device)

    avg_acc = 0
    for dataset_id, dataset_name in enumerate(args.dataset_order):
        logger.info("Start testing {}...".format(dataset_name))
        test_dataset = pickle.load(open(os.path.join(args.output_dir, 'test_dataset-{}'.format(dataset_id)), 'rb'))
        dataset_acc = test(dataset_id, args, model, test_dataset)
        avg_acc += dataset_acc / len(args.dataset_order)
    logger.info("Average acc: {:.3f}".format(avg_acc))


if __name__ == "__main__":
    main()
