import os
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from utils import tokenization, optimization, constants, misc
from utils.data import *

def get_classification_data(data_dir, data_name):
    """
    args:
        data_dir: str
        data_name: str
    return:
        data: dict of {"src_str": list of str, "lab": list of int}
    """
    src_0, src_1 = [], []
    with open(os.path.join(data_dir, data_name+".0"), 'r') as f:
        for line in f.readlines():
            src_0.append(line.strip())
    with open(os.path.join(data_dir, data_name+".1"), 'r') as f:
        for line in f.readlines():
            src_1.append(line.strip())
    lab_0 = [0] * len(src_0)
    lab_1 = [1] * len(src_1)
    
    src = src_0 + src_1
    lab = lab_0 + lab_1
    data = {"src_str": src, "lab": lab}
    print("%s data has been loaded" % data_name)
    for l, count in enumerate(np.bincount(data["lab"])):
        print("number of label %d: %d" % (l, count))
    return data

def load_and_cache_data(args, data_name, tokenizer):
    """
    return:
        data: dict of {"src_str": list of str,
                       "src_ind": list of int,
                       "lab": list of int}
    """
    sos_str = "_sos" if args.use_sos else ""
    eos_str = "_eos" if args.use_eos else ""
    mask_str = "_mask" if "mask" in args.vocab_file_name else ""
    cached_data_file = os.path.join(
        args.data_dir,
        f"cached_cls_{data_name}{sos_str}{eos_str}{mask_str}"
    )

    if os.path.exists(cached_data_file) and not args.overwrite_cache:
        print("Loading data from cached data file %s" % cached_data_file)
        data = torch.load(cached_data_file)
    else:
        print("Creating cached data file from data at %s" % cached_data_file)
        data = get_classification_data(args.data_dir, data_name)

        index_src = []
        str_src = []
        sos_id, eos_id = tokenizer.SOS_ID, tokenizer.EOS_ID
        sos_token, eos_token = tokenizer.SOS_TOKEN, tokenizer.EOS_TOKEN
        if args.use_sos and args.use_eos:
            for text in data['src_str']:
                index_src.append([sos_id] + tokenizer.encode(text) + [eos_id])
                str_src.append(' '.join([sos_token, text, eos_token]))
        elif args.use_sos:
            for text in data['src_str']:
                index_src.append([sos_id] + tokenizer.encode(text))
                str_src.append(' '.join([sos_token, text]))
        elif args.use_eos:
            for text in data['src_str']:
                index_src.append(tokenizer.encode(text) + [eos_id])
                str_src.append(' '.join([text, eos_token]))
        else:
            for text in data['src_str']:
                index_src.append(tokenizer.encode(text))
                str_src.append(text)

        data['src_ind'] = index_src
        data['src_str'] = str_src

        torch.save(data, cached_data_file)
    
    return data

class BasicTrainer:
    """
    Basic Trainer
    """
    def __init__(self, args, model, train_data=None, dev_data=None, test_data=None):
        self.args = args
        self.model = model
        self.optimizer = None
        self.scheduler = None

        self.train_dataloader = self.get_dataloader(train_data)\
            if train_data else None
        self.dev_dataloader = self.get_dataloader(dev_data)\
            if dev_data else None
        self.test_dataloader = self.get_dataloader(test_data)\
            if test_data else None

        if self.train_dataloader:
            self.optimizer, self.scheduler = self.get_optimizer()

    def get_dataloader(self, data):
        args = self.args
        if args.mode == "train":
            shuffle = args.shuffle
        else:
            shuffle = False
        dataset = ClassifierDataset(data["src_ind"], data["lab"])
        dataloader = DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=shuffle,
                                num_workers=args.num_workers,
                                collate_fn=ClassifierPaddingCollate)
        return dataloader
    
    def get_optimizer(self, params=None):
        args = self.args
        if params is None:
            params = self.model.parameters()
        train_dataloader = self.train_dataloader

        optimizer = optimization.get_optim(args, params)
        num_steps = len(train_dataloader) * args.num_train_epochs
        print("Total number of steps: %d" % num_steps)
        decay_step = len(train_dataloader) * args.decay_epoch
        if args.decay_epoch > 0:
            print("Step when lr starts to decay: %d" % decay_step)
            scheduler = optimization.get_constant_schedule_with_linear_decay(
                optimizer, decay_step=decay_step, num_training_steps=num_steps
            )
        else:
            scheduler = optimization.get_constant_schedule(optimizer)
        return optimizer, scheduler

    def save_checkpoint(self, path):
        # torch.save(self.args, os.path.join(path, "args.pt"))
        torch.save(self.model.state_dict(), os.path.join(path, "model_state_dict.pt"))
        # torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer_state_dict.pt"))
        # torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler_state_dict.pt"))
    
    def train(self, train_dataloader=None):
        print("\n### TRAINING BEGINS ###")
        args = self.args
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_dataloader = train_dataloader if train_dataloader else self.train_dataloader

        model.train()
        loss_record = []  # loss at global_step 0, 1, 2 ...
        acc_record = []
        global_step_record_for_eval = []
        global_step = -1

        model.zero_grad()

        start_time = time.time()
        for ep in range(args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                global_step += 1
                src, lab, src_len = batch
                src, lab = src.to(args.device), lab.to(args.device)

                try:
                    outputs = model(src)
                    loss = F.cross_entropy(outputs, lab, reduction='mean')
                    loss.backward()
                    g = clip_grad_norm(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    loss_record.append(loss.item())
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('|| WARNING: ran out of memory ||\n')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print('|| WARNING: fail to train ||\n')
                        raise e

                if global_step > 0 and global_step % args.log_interval == 0:
                    print(
                        f"epoch: {ep}  "\
                            f"step: {global_step}  "\
                            f"loss: {loss.item():.4f}  "\
                            f"||g||: {g:.2f}  "\
                            f"time: {misc.timeBetween(start_time, time.time())}"
                    )

                if global_step > 0 and global_step % args.eval_interval == 0:
                    print("\neval model at step: %d" % global_step)
                    acc = self.evaluate()
                    acc_record.append(acc)
                    global_step_record_for_eval.append(global_step)
                    checkpoint_output_dir = os.path.join(args.output_dir, "checkpoint-%d" % global_step)
                    if not os.path.exists(checkpoint_output_dir):
                        os.mkdir(checkpoint_output_dir)
                    print("Save checkpoint at %s" % checkpoint_output_dir)
                    self.save_checkpoint(checkpoint_output_dir)
                    model.train()
            
        print("### TRAINING ENDS ###\n")
        
        print("eval model at step: %d" % global_step)
        acc = self.evaluate()
        acc_record.append(acc)
        global_step_record_for_eval.append(global_step)
        checkpoint_output_dir = os.path.join(args.output_dir, "checkpoint-%d" % global_step)
        if not os.path.exists(checkpoint_output_dir):
            os.mkdir(checkpoint_output_dir)
        print("Save checkpoint at %s" % checkpoint_output_dir)
        self.save_checkpoint(checkpoint_output_dir)
        
        train_record = loss_record
        eval_record = (acc_record, global_step_record_for_eval)
        best_acc = self.save_train_result(train_record, eval_record)

        return best_acc, train_record, eval_record

    def evaluate(self, eval_dataloader=None):
        eval_dataloader = eval_dataloader if eval_dataloader else self.dev_dataloader
        args = self.args
        model = self.model

        model.eval()

        total_loss = 0
        total_preds, total_labs = [], []

        start_time = time.time()
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                src, lab, src_len = batch
                total_labs.extend(lab.numpy().tolist())
                src, lab = src.to(args.device), lab.to(args.device)

                try:
                    outputs = model(src)
                    total_loss += F.cross_entropy(outputs, lab, reduction='sum').item()
                    total_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('|| WARNING: ran out of memory ||\n')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print('|| WARNING: fail to train ||\n')
                        raise e
            acc = accuracy_score(total_labs, total_preds)
            print("==============================")
            print(
                "acc: {:.4f} loss: {:.4f} time: {}".format(
                    acc, total_loss/len(total_labs), misc.timeBetween(start_time, time.time())
                )
            )
            print("==============================\n")

        return acc

    def test(self, test_dataloader=None, save_res=None):
        test_dataloader = test_dataloader if test_dataloader else self.test_dataloader
        return self.evaluate(test_dataloader)

    def save_train_result(self, train_record, dev_record):
        args = self.args
        loss_record = train_record
        acc_record, gs_record = dev_record

        best_acc = np.max(acc_record)
        step_of_best_acc = gs_record[np.argmax(acc_record)]
        print("best acc: %.4f in step %d" % (best_acc, step_of_best_acc))

        with open(os.path.join(args.output_dir, "training_result.log"), 'w') as f:
            f.write("best acc: %.4f at step %d\n" % (best_acc, step_of_best_acc))
        
        plt.figure()
        plt.xlabel("step")
        plt.ylabel("acc")
        plt.plot(gs_record, acc_record)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "acc.pdf"), format='pdf')  # bbox_inches='tight'

        plt.figure()
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.plot(list(range(len(loss_record))), loss_record)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "loss.pdf"), format='pdf')

        return best_acc

class CNNClassifierTrainer(BasicTrainer):
    """
    CNN Classifier Trainer
    """
    def __init__(self, args, model, train_data=None, dev_data=None, test_data=None, **kwargs):
        super(CNNClassifierTrainer, self).__init__(
            args, model, train_data, dev_data, test_data
        )


class SelfAttnRNNClassifierTrainer(BasicTrainer):
    """
    Self-Attention RNN Classifier Trainer
    """
    def __init__(self, args, model, train_data=None, dev_data=None, test_data=None, **kwargs):
        super(SelfAttnRNNClassifierTrainer, self).__init__(
            args, model, train_data, dev_data, test_data
        )
        self.tokenizer = kwargs["tokenizer"]
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        if "mask" in args.vocab_file_name:
            self.args.mask_id = self.tokenizer.token2index["[mask]"]
            self.model.set_mask_id(self.args.mask_id)
    
    def train(self, train_dataloader=None):
        print("\n### TRAINING BEGINS ###")
        args = self.args
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_dataloader = train_dataloader if train_dataloader else self.train_dataloader

        model.train()
        loss_record = []  # loss at global_step 0, 1, 2 ...
        acc_record = []
        global_step_record_for_eval = []
        global_step = -1
        pad_id = args.pad_id

        model.zero_grad()
        if args.freeze_emb_at_beginning:
            model.freeze_emb()

        start_time = time.time()
        for ep in range(args.num_train_epochs):
            if ep == args.unfreeze_at_ep and args.freeze_emb_at_beginning:
                model.unfreeze_emb()
            for step, batch in enumerate(train_dataloader):
                global_step += 1
                src, lab, src_len = batch

                sorted_src_len, indices = torch.sort(src_len, dim=0, descending=True)
                sorted_src = torch.index_select(src, dim=0, index=indices)
                sorted_lab = torch.index_select(lab, dim=0, index=indices)

                sorted_src, sorted_src_len = sorted_src.to(args.device), sorted_src_len.to(args.device)
                sorted_lab = sorted_lab.to(args.device)

                try:
                    sorted_pad_mask = sorted_src == pad_id
                    sorted_outputs, _ = model(sorted_src, sorted_src_len, sorted_pad_mask)
                    loss = F.cross_entropy(sorted_outputs, sorted_lab, reduction='mean')
                    loss.backward()
                    g = clip_grad_norm(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    loss_record.append(loss.item())
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('|| WARNING: ran out of memory ||\n')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print('|| WARNING: fail to train ||\n')
                        raise e

                if global_step > 0 and global_step % args.log_interval == 0:
                    print(
                        f"epoch: {ep}  "\
                            f"step: {global_step}  "\
                            f"loss: {loss.item():.4f}  "\
                            f"||g||: {g:.2f}  "\
                            f"time: {misc.timeBetween(start_time, time.time())}"
                    )

                if global_step > 0 and global_step % args.eval_interval == 0:
                    print("\neval model at step: %d" % global_step)
                    acc = self.evaluate()
                    acc_record.append(acc)
                    global_step_record_for_eval.append(global_step)
                    checkpoint_output_dir = os.path.join(args.output_dir, "checkpoint-%d" % global_step)
                    if not os.path.exists(checkpoint_output_dir):
                        os.mkdir(checkpoint_output_dir)
                    print("Save checkpoint at %s" % checkpoint_output_dir)
                    self.save_checkpoint(checkpoint_output_dir)
                    model.train()
            
        print("### TRAINING ENDS ###\n")
        
        print("eval model at step: %d" % global_step)
        acc = self.evaluate()
        acc_record.append(acc)
        global_step_record_for_eval.append(global_step)
        checkpoint_output_dir = os.path.join(args.output_dir, "checkpoint-%d" % global_step)
        if not os.path.exists(checkpoint_output_dir):
            os.mkdir(checkpoint_output_dir)
        print("Save checkpoint at %s" % checkpoint_output_dir)
        self.save_checkpoint(checkpoint_output_dir)
        
        train_record = loss_record
        eval_record = (acc_record, global_step_record_for_eval)
        best_acc = self.save_train_result(train_record, eval_record)

        return best_acc, train_record, eval_record

    def evaluate(self, eval_dataloader=None):
        eval_dataloader = eval_dataloader if eval_dataloader else self.dev_dataloader
        args = self.args
        model = self.model

        model.eval()

        total_loss = 0
        total_preds, total_labs = [], []
        pad_id = args.pad_id

        start_time = time.time()
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                src, lab, src_len = batch
                total_labs.extend(lab.numpy().tolist())
                # src, lab = src.to(args.device), lab.to(args.device)

                sorted_src_len, indices = torch.sort(src_len, dim=0, descending=True)
                _, resorted_indices = torch.sort(indices, dim=0)
                sorted_src = torch.index_select(src, dim=0, index=indices)
                sorted_lab = torch.index_select(lab, dim=0, index=indices)

                sorted_src, sorted_src_len = sorted_src.to(args.device), sorted_src_len.to(args.device)
                sorted_lab = sorted_lab.to(args.device)
                resorted_indices = resorted_indices.to(args.device)

                try:
                    sorted_pad_mask = sorted_src == pad_id
                    sorted_outputs, _ = model(sorted_src, sorted_src_len, sorted_pad_mask)
                    total_loss += F.cross_entropy(sorted_outputs, sorted_lab, reduction='sum').item()
                    
                    outputs = torch.index_select(sorted_outputs, dim=0, index=resorted_indices)
                    total_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('|| WARNING: ran out of memory ||\n')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print('|| WARNING: fail to train ||\n')
                        raise e
            acc = accuracy_score(total_labs, total_preds)
            print("==============================")
            print(
                "acc: {:.4f} loss: {:.4f} time: {}".format(
                    acc, total_loss/len(total_labs), misc.timeBetween(start_time, time.time())
                )
            )
            print("==============================\n")

        return acc

    def test(self, test_dataloader=None, test_data=None, save_res=True):
        if test_dataloader is not None:
            assert test_data is not None
        else:
            test_dataloader = self.test_dataloader
            test_data = self.test_data
        test_lab = test_data['lab']
        test_src_str = test_data['src_str']

        args = self.args
        model = self.model

        model.eval()

        total_loss = 0
        total_preds = []
        # total_labs = []
        total_weights = []

        pad_id = args.pad_id

        start_time = time.time()
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                src, lab, src_len = batch
                # total_labs.extend(lab.numpy().tolist())

                sorted_src_len, indices = torch.sort(src_len, dim=0, descending=True)
                _, resorted_indices = torch.sort(indices, dim=0)
                sorted_src = torch.index_select(src, dim=0, index=indices)
                sorted_lab = torch.index_select(lab, dim=0, index=indices)

                sorted_src, sorted_src_len = sorted_src.to(args.device), sorted_src_len.to(args.device)
                sorted_lab = sorted_lab.to(args.device)
                resorted_indices = resorted_indices.to(args.device)

                try:
                    sorted_pad_mask = sorted_src == pad_id
                    sorted_outputs, sorted_weights = model(sorted_src, sorted_src_len, sorted_pad_mask)
                    total_loss += F.cross_entropy(sorted_outputs, sorted_lab, reduction='sum').item()
                    
                    outputs = torch.index_select(sorted_outputs, dim=0, index=resorted_indices)
                    weights = torch.index_select(sorted_weights, dim=0, index=resorted_indices)
                    
                    total_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
                    total_weights.extend(weights.cpu().numpy().tolist())

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('|| WARNING: ran out of memory ||\n')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print('|| WARNING: fail to train ||\n')
                        raise e
            acc = accuracy_score(test_lab, total_preds)
            print("==============================")
            print(
                "acc: {:.4f} loss: {:.4f} time: {}".format(
                    acc, total_loss/len(test_lab), misc.timeBetween(start_time, time.time())
                )
            )
            print("==============================\n")


            # print("type(test_src_str) is", type(test_src_str))  # <class 'list'>
            # print("type(test_src_str[0]) is", type(test_src_str[0]))  # <class 'str'>
            # print("type(test_lab) is", type(test_lab))  # <class 'list'>
            # print("type(test_lab[0]) is", type(test_lab[0]))  # <class 'int'>
            if save_res:
                with open(os.path.join(args.output_dir, "weights.csv"), 'w', newline='') as f:
                    csv_writer = csv.writer(f)
                    for ind, sample in enumerate(zip(
                        test_lab, total_preds, test_src_str, total_weights
                    )):
                        lab, pred, src_str, weights = sample
                        # src_str is a list
                        csv_writer.writerow([f"#{ind}"] + src_str.strip().split())
                        csv_writer.writerow([f"lab: {lab} (pred: {pred})"] + [f"{w:.2f}" for w in weights])

        return acc

    def gen_masked_src(self, data_name):
        if data_name == "train":
            dataloader = self.train_dataloader
            data = self.train_data
        elif data_name == "dev":
            dataloader = self.dev_dataloader
            data = self.dev_data
        elif data_name == "test":
            dataloader = self.test_dataloader
            data = self.test_data
        
        test_lab = data['lab']
        test_src_str = data['src_str']

        args = self.args
        model = self.model
        tokenizer = self.tokenizer

        model.eval()

        total_loss = 0
        total_preds = []
        # total_labs = []
        total_weights = []
        total_style_src = []
        total_content_src = []

        pad_id = args.pad_id

        start_time = time.time()
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                src, lab, src_len = batch
                # total_labs.extend(lab.numpy().tolist())

                sorted_src_len, indices = torch.sort(src_len, dim=0, descending=True)
                _, resorted_indices = torch.sort(indices, dim=0)
                sorted_src = torch.index_select(src, dim=0, index=indices)
                sorted_lab = torch.index_select(lab, dim=0, index=indices)

                sorted_src, sorted_src_len = sorted_src.to(args.device), sorted_src_len.to(args.device)
                sorted_lab = sorted_lab.to(args.device)
                resorted_indices = resorted_indices.to(args.device)

                try:
                    sorted_pad_mask = sorted_src == pad_id
                    sorted_outputs, sorted_weights, sorted_style_src,\
                        sorted_content_src = model.get_masked_src(sorted_src,
                        sorted_src_len, sorted_pad_mask)
                    total_loss += F.cross_entropy(sorted_outputs, sorted_lab, reduction='sum').item()
                    
                    outputs = torch.index_select(sorted_outputs, dim=0, index=resorted_indices)
                    weights = torch.index_select(sorted_weights, dim=0, index=resorted_indices)
                    style_src = torch.index_select(sorted_style_src, dim=0, index=resorted_indices)
                    content_src = torch.index_select(sorted_content_src, dim=0, index=resorted_indices)
                    
                    total_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
                    total_weights.extend(weights.cpu().numpy().tolist())
                    total_style_src.extend(style_src.cpu().numpy().tolist())
                    total_content_src.extend(content_src.cpu().numpy().tolist())

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('|| WARNING: ran out of memory ||\n')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print('|| WARNING: fail to train ||\n')
                        raise e
            acc = accuracy_score(test_lab, total_preds)
            print("==============================")
            print(
                "acc: {:.4f} loss: {:.4f} time: {}".format(
                    acc, total_loss/len(test_lab), misc.timeBetween(start_time, time.time())
                )
            )
            print("==============================\n")


            # print("type(test_src_str) is", type(test_src_str))  # <class 'list'>
            # print("type(test_src_str[0]) is", type(test_src_str[0]))  # <class 'str'>
            # print("type(test_lab) is", type(test_lab))  # <class 'list'>
            # print("type(test_lab[0]) is", type(test_lab[0]))  # <class 'int'>

            total_src_tokens = []
            total_style_tokens = []
            total_content_tokens = []

            with open(os.path.join(args.output_dir, f"{data_name}_weights.csv"), 'w', newline='') as f:
                csv_writer = csv.writer(f)
                for ind, sample in enumerate(zip(
                    test_lab, total_preds, test_src_str, total_weights,
                    total_style_src, total_content_src
                )):
                    lab, pred, src_str, weights, style_src, content_src = sample
                    # src_str is a list
                    src_tokens = src_str.strip().split()
                    csv_writer.writerow([f"#{ind}"] + list(map(lambda x:"'"+x, src_tokens)))
                    csv_writer.writerow([f"lab: {lab} (pred: {pred})"] + [f"{w:.2f}" for w in weights])

                    content_tokens = tokenizer.index_to_token(content_src, include_sos_eos=False)
                    style_tokens = tokenizer.index_to_token(style_src, include_sos_eos=False)

                    total_src_tokens.append(src_tokens[1:-1])
                    total_style_tokens.append(style_tokens)
                    total_content_tokens.append(content_tokens)

        return total_src_tokens, total_content_tokens, total_style_tokens



