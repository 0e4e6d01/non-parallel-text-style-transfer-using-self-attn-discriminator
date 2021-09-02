import os
import time
import csv
import pickle

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
from utils.evaluator import BLEUEvaluator


def get_transfer_data(data_dir, data_name):
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
    assert len(src) == len(lab)
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
        f"cached_transfer_{data_name}{sos_str}{eos_str}{mask_str}"
    )

    if os.path.exists(cached_data_file) and not args.overwrite_cache:
        print("Loading data from cached data file %s" % cached_data_file)
        data = torch.load(cached_data_file)
    else:
        print("Creating cached data file from data at %s" % cached_data_file)
        data = get_transfer_data(args.data_dir, data_name)

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

def lambda_schedule(num_iter, start=0.0, stop=1.0, ratio=0.1):
    lambdas = np.ones(num_iter) * stop
    progress_interval = num_iter * ratio
    for i in range(int(progress_interval)):
        lambdas[i] *= i / progress_interval

    return lambdas

class BasicTrainer:
    """
    Basic Trainer
    """
    def __init__(self, args, model, train_data=None, dev_data=None, test_data=None,
            tokenizer=None):
        self.args = args
        self.model = model
        self.optimizer = None
        self.scheduler = None

        self.train_dataloader = self.get_dataloader(train_data, "train")\
            if train_data else None
        self.dev_dataloader = self.get_dataloader(dev_data, "dev")\
            if dev_data else None
        self.test_dataloader = self.get_dataloader(test_data, "test")\
            if test_data else None

        if self.train_dataloader:
            self.optimizer, self.scheduler = self.get_optimizer()

    def get_dataloader(self, data, data_name):
        args = self.args
        if data_name == "train":
            shuffle = args.shuffle
            batch_size = args.batch_size
        else:
            shuffle = False
            # batch_size = 2
            batch_size = args.batch_size
        dataset = ClassifierDataset(data["src_ind"], data["lab"])
        dataloader = DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=shuffle,
                                num_workers=args.num_workers,
                                collate_fn=ClassifierPaddingCollate)
        return dataloader
    
    def get_optimizer(self):
        args = self.args
        model = self.model
        train_dataloader = self.train_dataloader

        optimizer = optimization.get_optim(args, model.parameters())
        num_steps = len(train_dataloader) * args.num_train_epochs
        args.num_steps = num_steps
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
        return
    
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()
    
    def test(self):
        raise NotImplementedError()

    def save_train_result(self, train_record, eval_record):
        args = self.args
        train_loss_record = train_record
        eval_bleu_record, eval_gs_record = eval_record

        best_bleu = np.max(eval_bleu_record)
        step_of_best_bleu = eval_gs_record[np.argmax(eval_bleu_record)]
        print("best BLEU: %.4f in step %d" % (best_bleu, step_of_best_bleu))

        with open(os.path.join(args.output_dir, "training_result.log"), 'w') as f:
            f.write("best BLEU: %.4f in step %d" % (best_bleu, step_of_best_bleu))
        
        plt.figure()
        plt.xlabel("step")
        plt.ylabel("BLEU")
        plt.plot(eval_gs_record, eval_bleu_record)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "bleu.pdf"), format='pdf')  # bbox_inches='tight'

        plt.figure()
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.plot(list(range(len(train_loss_record))), train_loss_record)
        # plt.plot(eval_gs_record, eval_loss_record)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "loss.pdf"), format='pdf')

        return best_bleu, step_of_best_bleu

class TransferModelTrainer(BasicTrainer):

    def __init__(self, args, model, train_data=None, dev_data=None,
        test_data=None, **kwargs):
        super().__init__(
            args, model, train_data, dev_data, test_data
        )
        self.tokenizer = kwargs["tokenizer"]

        if self.args.cls_model_path:
            print(f"Load classifier model form {self.args.cls_model_path}")
            self.model.classifier.load_state_dict(
                torch.load(
                    os.path.join(self.args.cls_model_path, "model_state_dict.pt")
                )
            )
            self.model.freeze_cls()

        # args.cls_weight = 0.05
        # args.ca_weight = 0.0
        # args.bt_weight = 1.0
        self.use_caw_schedule = False

        del self.optimizer
        del self.scheduler

        if self.train_dataloader:
            params = []
            for k, v in self.model.named_parameters():
                # print("%s: %s" % (k, str(v.shape)))
                if "classifier" in k or "lm" in k:
                    print("not optimize %s" % k)
                else:
                    print("add params of %s to optimizer" % k)
                    params.append(v)
            self.optimizer, self.scheduler\
                = self.get_optimizer(params)
        
        # torch.autograd.set_detect_anomaly(True)

        self.clf_model = torch.load(args.cnn_clf_path).to(args.device)
        self.clf_model.eval()

        self.dev_ref_path_list = getattr(args, "dev_ref_path_list", None)
        self.test_ref_path_list = getattr(args, "test_ref_path_list", None)
        if self.test_ref_path_list is None:
            self.test_ref_path_list = self.args.ref_list
        print("self.dev_ref_path_list is")
        print(self.dev_ref_path_list)
        print("self.test_ref_path_list is")
        print(self.test_ref_path_list)

        if not self.args.use_bpe:
            self.dev_data_path_list = [
                [os.path.join(self.args.data_dir, f"dev.{i}")] for i in range(2)
            ]
            self.test_data_path_list = [
                [os.path.join(self.args.data_dir, f"test.{i}")] for i in range(2)
            ]
        else:
            self.dev_data_path_list = [
                [os.path.join(self.args.data_dir, f"self_ref.dev.{i}")] for i in range(2)
            ]
            self.test_data_path_list = [
                [os.path.join(self.args.data_dir, f"self_ref.test.{i}")] for i in range(2)
            ]
        
        print("self.dev_data_path_list is")
        print(self.dev_data_path_list)
        print("self.test_data_path_list is")
        print(self.test_data_path_list)

    def get_optimizer(self, params=None):
        args = self.args
        if params is None:
            print("return because params is None")
            return None, None
            # params = self.model.parameters()
        train_dataloader = self.train_dataloader

        optimizer = optimization.get_optim(args, params)
        num_steps = len(train_dataloader) * args.num_train_epochs // args.grad_accum_interval
        args.num_steps = num_steps
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

    def train(self, train_dataloader=None):

        print("\n### TRAINING BEGINS ###")
        args = self.args
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_dataloader = train_dataloader if train_dataloader else self.train_dataloader

        model.train()
        loss_record = []  # loss at global_step 0, 1, 2 ...
        dev_metric_record = []
        global_step_record_for_eval = []
        global_step = 0
        pad_id = args.pad_id

        grad_accum_interval = args.grad_accum_interval
        log_loss = 0.0
        num_iters_per_epoch = len(train_dataloader)
        normalizer = min(num_iters_per_epoch, grad_accum_interval)

        cls_w = args.cls_weight
        print("cls_w is", cls_w)
        if self.use_caw_schedule:
            start = 0.0
            stop = args.ca_weight
            ratio = 0.5
            ca_w_list = lambda_schedule(args.num_steps,
                start=start, stop=stop, ratio=ratio)
            print(f"ca_w uses schedule (start={start}, stop={stop}, ratio={ratio})")
            ca_w = ca_w_list[0]
        else:
            ca_w = args.ca_weight
            print("ca_w is", ca_w)
        bt_w = args.bt_weight
        print("bt_w is", bt_w)

        model.zero_grad()
        if args.freeze_emb_at_beginning:
            model.freeze_emb()
        
        start_time = time.time()
        for ep in range(args.num_train_epochs):
            if ep == args.unfreeze_at_ep and args.freeze_emb_at_beginning:
                model.unfreeze_emb()
            for step, batch in enumerate(train_dataloader):
                src, lab, src_len = batch
                # print(f"ep:{ep}, step: {step}, src.shape[1] is", src.shape[1])
                sorted_src_len, indices = torch.sort(src_len, dim=0, descending=True)
                sorted_src = torch.index_select(src, dim=0, index=indices)
                sorted_lab = torch.index_select(lab, dim=0, index=indices)
                
                sorted_src = sorted_src.to(args.device)
                sorted_src_len = sorted_src_len.to(args.device)
                sorted_lab = sorted_lab.to(args.device)

                try:
                    sorted_src_pad_mask = sorted_src==pad_id
                    sorted_loss_tuple, sorted_output_tuple,\
                        sorted_algin = model(sorted_src, sorted_src_len,
                        sorted_lab, sorted_src_pad_mask)
                    sorted_rec_loss, sorted_bt_loss,\
                        sorted_src_cls_loss, sorted_soft_out_cls_loss,\
                        sorted_out_cls_loss, sorted_ca_loss = sorted_loss_tuple
                    sorted_output, sorted_output_len = sorted_output_tuple
                    
                    rec_loss = sorted_rec_loss.mean()
                    bt_loss = sorted_bt_loss.mean()
                    src_cls_loss = sorted_src_cls_loss.mean()
                    soft_out_cls_loss = sorted_soft_out_cls_loss.mean()
                    out_cls_loss = sorted_out_cls_loss.mean()
                    ca_loss = sorted_ca_loss.mean()
                    loss = rec_loss + bt_w * bt_loss\
                        + cls_w * soft_out_cls_loss + ca_w * ca_loss
                    loss /= normalizer
                    loss.backward()

                    if (step+1) % grad_accum_interval == 0 or\
                        (grad_accum_interval >= num_iters_per_epoch and
                            (step+1) == num_iters_per_epoch):
                        g = clip_grad_norm(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                        loss_record.append(log_loss)
                        # global_step += 1
                        log_loss = 0.0
                        if global_step > 0 and global_step % args.log_interval == 0:
                            
                            print(
                                f"epoch: {ep}  "\
                                    f"step: {global_step}  "\
                                    f"loss: {loss.item() * normalizer:.4f}  "\
                                    f"rec_loss: {rec_loss.item():.4f}  "\
                                    f"bt_loss: {bt_loss.item():.4f}  "\
                                    f"src_cls_loss: {src_cls_loss.item():.4f}  "\
                                    f"soft_out_cls_loss: {soft_out_cls_loss.item():.4f}  "\
                                    f"out_cls_loss: {out_cls_loss.item():.4f}  "\
                                    f"ca_loss: {ca_loss.item():.4f}  "\
                                    f"||g||: {g:.2f}  "\
                                    f"ca_w: {ca_w:.4f}  "\
                                    f"time: {misc.timeBetween(start_time, time.time())}"
                            )

                        if global_step > 0 and global_step % args.eval_interval == 0:
                            print("\neval model at step: %d" % global_step)
                            checkpoint_output_dir = os.path.join(args.output_dir, "checkpoint-%d" % global_step)
                            if not os.path.exists(checkpoint_output_dir):
                                os.mkdir(checkpoint_output_dir)
                            org_output_dir = args.output_dir
                            args.output_dir = checkpoint_output_dir
                            print("dev")
                            dev_metric = self.evaluate()
                            dev_metric_record.append(dev_metric)
                            global_step_record_for_eval.append(global_step)
                            args.output_dir = org_output_dir
                            print("Save checkpoint at %s" % checkpoint_output_dir)
                            self.save_checkpoint(checkpoint_output_dir)
                            model.train()

                        global_step += 1
                        if self.use_caw_schedule:
                            ca_w = ca_w_list[global_step]
                    else:
                        log_loss += loss.item()
                        
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('|| WARNING: ran out of memory ||\n')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print('|| WARNING: fail to train ||\n')
                        raise e
                    raise e
        # gpu_profile(frame=sys._getframe(), event='line', arg=None)
            
        print("### TRAINING ENDS ###\n")
        
        print("\neval model at step: %d" % global_step)
        checkpoint_output_dir = os.path.join(args.output_dir, "checkpoint-%d" % global_step)
        if not os.path.exists(checkpoint_output_dir):
            os.mkdir(checkpoint_output_dir)
        org_output_dir = args.output_dir
        args.output_dir = checkpoint_output_dir
        print("dev")
        dev_metric = self.evaluate()
        dev_metric_record.append(dev_metric)
        global_step_record_for_eval.append(global_step)
        args.output_dir = org_output_dir
        print("Save checkpoint at %s" % checkpoint_output_dir)
        self.save_checkpoint(checkpoint_output_dir)
        
        train_record = loss_record
        eval_record = (dev_metric_record, global_step_record_for_eval)
        with open(os.path.join(args.output_dir, "record.pt"), "wb") as f:
            pickle.dump({"train": train_record, "eval": eval_record}, f)
        
        self.save_train_result(train_record, eval_record)

        return train_record, eval_record

    def evaluate(self, eval_dataloader=None, data_path_list=None, ref_path_list=None, data_name="dev"):
        eval_dataloader = eval_dataloader if eval_dataloader else self.dev_dataloader
        ref_path_list = ref_path_list if ref_path_list else self.dev_ref_path_list
        data_path_list = data_path_list if data_path_list else self.dev_data_path_list
        args = self.args
        model = self.model
        tokenizer = self.tokenizer
        clf_model = self.clf_model

        model.eval()

        num_data = 0
        total_loss = 0
        total_rec_loss = 0
        total_bt_loss = 0
        total_src_cls_loss = 0
        total_soft_out_cls_loss = 0
        total_out_cls_loss = 0
        total_ca_loss = 0

        outputs_list = []
        outputs_len_list = []
        lab_list = []
        clf_preds_list = []

        cls_w = args.cls_weight
        ca_w = args.ca_weight
        bt_w = args.bt_weight

        pad_id = args.pad_id

        start_time = time.time()
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                src, lab, src_len = batch
                num_data += src.shape[0]
                # print(f"ep:{ep}, step: {step}, src.shape[1] is", src.shape[1])
                sorted_src_len, indices = torch.sort(src_len, dim=0, descending=True)
                _, resorted_indices = torch.sort(indices, dim=0)
                sorted_src = torch.index_select(src, dim=0, index=indices)
                sorted_lab = torch.index_select(lab, dim=0, index=indices)
                
                sorted_src = sorted_src.to(args.device)
                sorted_src_len = sorted_src_len.to(args.device)
                sorted_lab = sorted_lab.to(args.device)
                resorted_indices = resorted_indices.to(args.device)

                try:
                    sorted_src_pad_mask = sorted_src==pad_id
                    sorted_loss_tuple, sorted_outputs_tuple,\
                        sorted_algin = model(sorted_src, sorted_src_len,
                        sorted_lab, sorted_src_pad_mask)
                    sorted_rec_loss, sorted_bt_loss,\
                        sorted_src_cls_loss, sorted_soft_out_cls_loss,\
                        sorted_out_cls_loss, sorted_ca_loss = sorted_loss_tuple
                    sorted_outputs, sorted_outputs_len = sorted_outputs_tuple
                    # shape of sorted_outputs is [batch_size, max_len]
                    outputs = torch.index_select(sorted_outputs, dim=0, index=resorted_indices)
                    outputs_len = torch.index_select(sorted_outputs_len, dim=0, index=resorted_indices)

                    clf_preds = torch.argmax(clf_model(outputs), dim=-1)

                    rec_loss = sorted_rec_loss.sum()
                    bt_loss = sorted_bt_loss.sum()
                    src_cls_loss = sorted_src_cls_loss.sum()
                    soft_out_cls_loss = sorted_soft_out_cls_loss.sum()
                    out_cls_loss = sorted_out_cls_loss.sum()
                    ca_loss = sorted_ca_loss.sum()
                    loss = rec_loss + bt_w * bt_loss\
                        + cls_w * soft_out_cls_loss + ca_w * ca_loss

                    total_rec_loss += rec_loss.item()
                    total_bt_loss += bt_loss.item()
                    total_src_cls_loss += src_cls_loss.item()
                    total_soft_out_cls_loss += soft_out_cls_loss.item()
                    total_out_cls_loss += out_cls_loss.item()
                    total_ca_loss += ca_loss.item()
                    total_loss += loss.item()

                    outputs_list.extend(
                        [x.squeeze(0) for x in torch.split(outputs, split_size_or_sections=1, dim=0)]
                    )
                    outputs_len_list.extend(
                        [x.squeeze(0) for x in torch.split(outputs_len, split_size_or_sections=1, dim=0)]
                    )
                    lab_list.extend(
                        [x.squeeze(0) for x in torch.split(lab, split_size_or_sections=1, dim=0)]
                    )
                    clf_preds_list.extend(
                        [x.squeeze(0).item() for x in torch.split(clf_preds, split_size_or_sections=1, dim=0)]
                    )

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('|| WARNING: ran out of memory ||\n')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print('|| WARNING: fail to train ||\n')
                        raise e
                    
            eval_loss = total_loss / num_data
            eval_rec_loss = total_rec_loss / num_data
            eval_bt_loss = total_bt_loss / num_data
            eval_src_cls_loss = total_src_cls_loss / num_data
            eval_soft_out_cls_loss = total_soft_out_cls_loss / num_data
            eval_out_cls_loss = total_out_cls_loss / num_data
            eval_ca_loss = total_ca_loss / num_data

            inv_lab_list = 1-np.array(lab_list)
            # print("clf_preds_list is")
            # print(clf_preds_list)
            eval_acc = accuracy_score(inv_lab_list, np.array(clf_preds_list)) * 100.0
            
            transfer_file_names = [
                os.path.join(args.output_dir, f"{data_name}.0.tsf"),
                os.path.join(args.output_dir, f"{data_name}.1.tsf")
            ]
            transfer_files = [
                open(transfer_file_names[0], 'w'),
                open(transfer_file_names[1], 'w')
            ]
            count = 0
            # print(f"len(outputs_list): {len(outputs_list)}, len(outputs_len_list): {len(outputs_len_list)}")
            
            for output, output_len, l in zip(outputs_list, outputs_len_list, lab_list):
                # print("output is", output)
                text = tokenizer.decode(output, include_sos_eos=False)
                if output_len < args.max_decoding_len:
                    pass
                if args.use_bpe:
                    text = text.replace("@@ ", "")
                    text = text.strip("@@")
                transfer_files[l].write(text+'\n')
                count += 1
            transfer_files[0].close()
            transfer_files[1].close()

            try:
                assert count == num_data
            except:
                print(f"count: {count}, total_num: {num_data}")
                raise RuntimeError()

            bleu_evaluator = BLEUEvaluator()

            if ref_path_list is not None:
                bleu_score_021 = bleu_evaluator.score(ref_path_list[0], transfer_file_names[0])
                bleu_score_120 = bleu_evaluator.score(ref_path_list[1], transfer_file_names[1])
                bleu_score = (bleu_score_021 + bleu_score_120) / 2
            else:
                bleu_score = None
            
            if data_path_list is not None:
                self_bleu_score_021 = bleu_evaluator.score(data_path_list[0], transfer_file_names[0])
                self_bleu_score_120 = bleu_evaluator.score(data_path_list[1], transfer_file_names[1])
                self_bleu_score = (self_bleu_score_021 + self_bleu_score_120) / 2
            else:
                self_bleu_score = None

            print("==============================")
            if ref_path_list is not None:
                print(
                    f"BLEU: {bleu_score:.4f}  "\
                        f"(0->1:{bleu_score_021:.4f}, 1->0:{bleu_score_120:.4f})  ",
                    end='',
                )
            if data_path_list is not None:
                print(
                    f"self-BLEU: {self_bleu_score:.4f}  "\
                        f"(0->1:{self_bleu_score_021:.4f}, 1->0:{self_bleu_score_120:.4f})  ",
                    end='',
                )
            print(
                f"acc: {eval_acc:.4f}\n"\
                    f"loss: {eval_loss:.4f}  "\
                    f"rec_loss: {eval_rec_loss:.4f}  "\
                    f"bt_loss: {eval_bt_loss:.4f}  "\
                    f"src_cls_loss: {eval_src_cls_loss:.4f}  "\
                    f"soft_out_cls_loss: {eval_soft_out_cls_loss:.4f}  "\
                    f"out_cls_loss: {eval_out_cls_loss:.4f}  "\
                    f"ca_loss: {eval_ca_loss:.4f}  "\
                    f"time: {misc.timeBetween(start_time, time.time())}"
            )
            print("==============================\n")

        return (bleu_score, self_bleu_score, eval_acc)

    def test(self, test_dataloader=None, data_path_list=None, ref_path_list=None):
        test_dataloader = test_dataloader if test_dataloader else self.test_dataloader
        ref_path_list = ref_path_list if ref_path_list else self.test_ref_path_list
        data_path_list = data_path_list if data_path_list else self.test_data_path_list
        return self.evaluate(test_dataloader, data_path_list, ref_path_list, "test")
    
    def save_train_result(self, train_record, eval_record):
        args = self.args
        train_loss_record = train_record
        dev_metric_record, eval_gs_record = eval_record

        dev_unzip = list(zip(*dev_metric_record))
        dev_bleu_record, dev_self_bleu_record, dev_acc_record = np.array(dev_unzip[0]),\
            np.array(dev_unzip[1]), np.array(dev_unzip[2])

        if (dev_bleu_record!=None).all():
            best_dev_bleu = np.max(dev_bleu_record)
            step_of_best_dev_bleu = eval_gs_record[np.argmax(dev_bleu_record)]
            print("best dev BLEU: %.4f in step %d" % (best_dev_bleu, step_of_best_dev_bleu))

        fig = plt.figure()
        ax_1 = fig.add_subplot(111)
        ax_2 = ax_1.twinx()

        ax_1.set_xlabel("step")
        ax_1.set_ylabel("(self-)BLEU")
        ax_2.set_ylabel("Acc")

        line_list = []
        line_label_list = []

        if (dev_bleu_record!=None).all():
            # l, = ax_1.plot(eval_gs_record, dev_bleu_record, '-', c='#1f77b4', label="dev BLEU")
            l, = ax_1.plot(eval_gs_record, dev_bleu_record, '-', c='#1f77b4')
            line_list.append(l)
            line_label_list.append("dev BLEU")

        # l, = ax_1.plot(eval_gs_record, dev_self_bleu_record, ':', c='#1f77b4', label="dev self-BLEU")
        l, = ax_1.plot(eval_gs_record, dev_self_bleu_record, ':', c='#1f77b4')
        line_list.append(l)
        line_label_list.append("dev self-BLEU")

        # l, = ax_2.plot(eval_gs_record, dev_acc_record, '--', c='#1f77b4', label="dev acc")
        l, = ax_2.plot(eval_gs_record, dev_acc_record, '--', c='#1f77b4')
        line_list.append(l)
        line_label_list.append("dev acc")


        plt.legend(line_list, line_label_list)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "bleu_and_acc.pdf"), format='pdf')  # bbox_inches='tight'
        plt.close()

        plt.figure()
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.plot(list(range(len(train_loss_record))), train_loss_record)
        # plt.plot(eval_gs_record, eval_loss_record)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "loss.pdf"), format='pdf')
        plt.close()


