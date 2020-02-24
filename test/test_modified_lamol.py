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
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from settings import parse_test_args, model_classes, init_logging
from utils_class import TextClassificationDataset, dynamic_collate_fn, prepare_inputs, DynamicBatchSampler
from ipdb import set_trace as bp


def local_adapt(input_ids, label, tmp_model, q_input_ids, q_masks, q_labels, args, org_params):

    q_input_ids = q_input_ids.to(args.devices[0]).detach()
    q_masks = q_masks.to(args.devices[0]).detach()
    q_labels = q_labels.to(args.devices[0]).detach()

    # optimizer = optim.SGD(tmp_model.parameters(), lr=args.adapt_lr, momentum=0.9)
    optimizer = torch.optim.Adam(tmp_model.parameters(), lr=args.adapt_lr, eps=args.adam_epsilon)

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
        # bp()
        loss, logits = predict(tmp_model, input_ids, label)
        losses.append(loss)
        accs.append(int(np.argmax(logits, axis=1) == label.detach().cpu().numpy()))
        if step == args.adapt_steps:
            break
        # make 1 local step
        tmp_model.train()
        params = torch.cat([torch.reshape(param, [-1]) for param in tmp_model.parameters()], 0)
        from math import ceil
        num_batch = ceil(args.n_neighbors / args.batch_size)
        for i in range(num_batch):
            # bp()
            lower = i * args.batch_size
            upper = (i + 1) * args.batch_size
            output = tmp_model(input_ids=q_input_ids[lower:upper], attention_mask=q_masks[lower:upper], labels=q_labels[lower:upper])[:2]
            loss = output[0] + args.adapt_lambda * torch.sum((org_params - params) ** 2)
            tmp_model.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(tmp_model.parameters(), args.max_grad_norm)
            optimizer.step()
        
    assert len(accs) == len(losses) == args.adapt_steps + 1
    return accs, losses


def plot_acc_and_loss(accs, losses, file_name):
    assert len(accs) == len(losses)
    mean_losses = np.mean(losses, axis=0)
    mean_acces = np.mean(accs, axis=0)
    local_steps = np.arange(0, len(mean_acces))
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(local_steps, mean_losses, marker='o')
    ax2.plot(local_steps, mean_acces, marker='o')
    ax1.set_ylabel("loss")
    ax2.set_xlabel("no. of adapt_steps")
    ax2.set_ylabel("acc")
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f'ntest{len(accs)}')
    plt.savefig(f"{file_name}_ntest{len(accs)}.png")


def test_task(task_id, args, model, test_dataset):
    # bp() 
    if args.fp16_test:
        model = model.half()
    
    def update_metrics(loss, logits, cur_loss, cur_acc):
        preds = np.argmax(logits, axis=1)
        return cur_loss + loss, cur_acc + np.sum(preds == labels.detach().cpu().numpy())

    test_size = args.n_test
    cur_loss, cur_acc = 0, 0
    if args.adapt_steps >= 1:
        cur_accs = []
        cur_losses = []
        with torch.no_grad():
            org_params = torch.cat([torch.reshape(param, [-1]) for param in model.parameters()], 0)

        q_input_ids = pickle.load(open(os.path.join(args.output_dir, 'q_input_ids-{}'.format(task_id)), 'rb'))
        q_masks = pickle.load(open(os.path.join(args.output_dir, 'q_masks-{}'.format(task_id)), 'rb'))
        q_labels = pickle.load(open(os.path.join(args.output_dir, 'q_labels-{}'.format(task_id)), 'rb'))

        for i in range(test_size):
            labels, input_ids = test_dataset[i]
            labels = torch.tensor(np.expand_dims(labels, 0), dtype=torch.long).to(args.devices[0])
            input_ids = torch.tensor(np.expand_dims(input_ids, 0), dtype=torch.long).to(args.devices[0])
            accs, losses = local_adapt(input_ids, labels, copy.deepcopy(model), q_input_ids[i], q_masks[i], q_labels[i], args, org_params)
            if len(cur_accs) == len(cur_losses) == 0:
                cur_losses = np.array([losses])
                cur_accs = np.array([accs])
            else:
                cur_losses = np.concatenate([cur_losses, [losses]])
                cur_accs = np.concatenate([cur_accs, [accs]])
            
            if (i+1) % args.logging_steps == 0:
                logging.info("Local adapted {}/{} examples, test loss: {:.3f} , test acc: {:.3f}".format(
                    i+1, test_size, cur_losses.mean(axis=0)[-1],
                    cur_accs.mean(axis=0)[-1]))
        pickle.dump({"accuracy": cur_accs, "loss": cur_losses},
                open(f"{args.output_dir}/lr{args.adapt_lr}_metrics_against_adapt_step_{task_id}_ntest{args.n_test}_step{args.adapt_steps}", "wb"))
        plot_acc_and_loss(accs=cur_accs, losses=cur_losses,
                file_name=f"{args.output_dir}/lr{args.adapt_lr}_metrics_against_adapt_step_plot_{task_id}_ntest{args.n_test}_step{args.adapt_steps}")
        logger.info("test loss: {:.3f} , test acc: {:.3f}".format(
            cur_losses.mean(axis=0)[-1], cur_accs.mean(axis=0)[-1]))
        return cur_accs.mean(axis=0), cur_losses.mean(axis=0)

    else:
        test_dataloader = DataLoader(test_dataset, num_workers=args.n_workers, collate_fn=dynamic_collate_fn,
                                     batch_sampler=DynamicBatchSampler(test_dataset, args.batch_size))
        tot_n_inputs = 0
        for step, batch in enumerate(test_dataloader):
            n_inputs, input_ids, masks, labels = prepare_inputs(args, batch)
            tot_n_inputs += n_inputs
            # bp()
            with torch.no_grad():
                model.eval()
                outputs = model(input_ids=input_ids, attention_mask=masks, labels=labels)
                loss = outputs[0].item()
                logits = outputs[1].detach().cpu().numpy()
            # bp()
            cur_loss, cur_acc = update_metrics(loss * n_inputs, logits, cur_loss, cur_acc)
            if (step+1) % args.logging_steps == 0:
                logging.info("Tested {}/{} examples , test loss: {:.3f} , test acc: {:.3f}".format(
                    tot_n_inputs, len(test_dataset), cur_loss/tot_n_inputs, cur_acc/tot_n_inputs))
        assert tot_n_inputs == len(test_dataset)
        # bp()

        logger.info("test loss: {:.3f} , test acc: {:.3f}".format(
            cur_loss / test_size, cur_acc / test_size))
        return cur_loss / test_size, cur_acc / test_size


def main():
    args = parse_test_args()
    train_args = pickle.load(open(os.path.join(args.output_dir, 'train_args'), 'rb'))
    # assert train_args.output_dir == args.output_dir
    train_args.__dict__.update(args.__dict__)
    args = train_args
    init_logging(os.path.join(args.output_dir, args.test_log_filename))
    logger.info("args: " + str(args))

    config_class, model_class, args.tokenizer_class = model_classes[args.model_type]
    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    save_model_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(len(args.tasks)-1))
    model = model_class.from_pretrained(save_model_path, config=model_config).to(args.devices[0])

    avg_accs = []
    avg_losses = []
    # bp()
    for task_id, task in enumerate(args.tasks):
        logger.info("Start testing {}...".format(task))
        test_dataset = pickle.load(open(os.path.join(args.output_dir, 'test_dataset-{}'.format(task_id)), 'rb'))
        task_loss, task_acc = test_task(task_id, args, model, test_dataset)
        avg_accs.append(task_acc)
        avg_losses.append(task_loss)
        # bp()
    avg_accs = np.array(avg_accs)
    avg_losses = np.array(avg_losses)

    # plot_acc_and_loss(accs=avg_accs, losses=avg_losses,
      #        file_name=f"{args.output_dir}/lr{args.adapt_lr}_metrics_against_adapt_step_plot_avg_ntest{args.n_test}_steps{args.adapt_steps}")
    # logger.info("Average acc: {:.3f}".format(avg_accs.mean(axis=0)[-1]))
    # logger.info("Average loss: {:.3f}".format(avg_losses.mean(axis=0)[-1]))


if __name__ == "__main__":
    main()
