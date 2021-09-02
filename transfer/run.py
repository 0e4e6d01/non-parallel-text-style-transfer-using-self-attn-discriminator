import os
import sys
import time
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

sys.path.append(".")
# sys.path.append("..")
from common_args import add_common_arguments
from transfer.transfer_args import *
import transfer.models as models
import transfer.trainers as trainers
from utils import tokenization, optimization, constants, misc
from utils.data import ClassifierDataset, ClassifierPaddingCollate
from utils.evaluator import BLEUEvaluator

def load_arguments():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    add_transfer_arguments(parser)
    add_rnn_arguments(parser)
    add_cnn_arguments(parser)
    args = parser.parse_args()

    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(os.path.join(path, "config.yaml"), 'r') as f:
        args = yaml.load(f)
    return args

def save_config(args):
    with open(os.path.join(args.output_dir, "config.yaml"), 'w') as f:
        yaml.dump(args, f)

def save_tokenizer(args, tokenizer):
    torch.save(tokenizer, os.path.join(args.output_dir, "tokenizer.pt"))

def load_embedding(path):
    raise NotImplementedError("function load_embedding has not been implemented.")

def get_fn_init_weight(init_fn):
    
    def init_weights(m):
        # print(type(m))
        if 'Embedding' in str(type(m)):
            # print('Embedding')
            return
        if 'LSTM' in str(type(m)) or 'GRU' in str(type(m)):
            # print('RNN')
            init_fn(m.weight_ih_l0)
            init_fn(m.weight_hh_l0)
            try:
                init_fn(m.weight_ih_l0_reverse)
                init_fn(m.weight_hh_l0_reverse)
            except:
                pass
        elif 'Linear' in str(type(m)):
            # print('Linear')
            init_fn(m.weight)

    return init_weights
    
def get_model(args, embedding):
    if embedding is not None:
        print("using pre-trained embedding")
    model = getattr(models, args.model_name)(
        config=args,
        embedding=embedding,
    )
    
    if args.model_path:
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_path, "model_state_dict.pt")
            )
        )
    else:
        if args.init == 'xavier_uniform':
            print('Use xavier_uniform')
            model.apply(get_fn_init_weight(nn.init.xavier_uniform_))
        elif args.init == 'xavier_normal':
            print('Use xavier_normal')
            model.apply(get_fn_init_weight(nn.init.xavier_normal_))
        elif args.init == 'kaiming_uniform':
            print('Use kaiming_uniform')
            model.apply(get_fn_init_weight(nn.init.kaiming_uniform_))
        elif args.init == 'kaiming_normal':
            print('Use kaiming_normal')
            model.apply(get_fn_init_weight(nn.init.kaiming_normal_))
        else:
            print('Invalid init method %s, use default init method instead' % args.init)
    
    model.to(args.device)

    param_count = 0
    for param in model.parameters():  # param is a tensor
        param_count += param.view(-1).size()[0]
    
    print("")
    print(repr(model))
    print('Total number of parameters: %d\n' % param_count)
    
    return model


def get_trainer(args, model, train_data=None, dev_data=None, test_data=None, tokenizer=None):
    trainer = getattr(trainers, args.model_name+"Trainer")(
        args=args,
        model=model,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data,
        tokenizer=tokenizer,
    )
    return trainer

def get_references(ref_dir):
    file_list = os.listdir(ref_dir)
    ref_list = [[], []]
    for f in file_list:
        suffix = f.split('.')[-1]
        if suffix == '0':
            ref_list[0].append(os.path.join(ref_dir, f))
        elif suffix == '1':
            ref_list[1].append(os.path.join(ref_dir, f))
    return ref_list

def main(args):

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("args.device is", args.device)
    if args.dev_ref_dir is not None:
        args.dev_ref_path_list = get_references(args.dev_ref_dir)
    else:
        args.dev_ref_path_list = None

    if args.test_ref_dir is not None:
        args.test_ref_path_list = get_references(args.test_ref_dir)
    else:
        args.test_ref_path_list = None

    print("dev_ref_path_list:")
    print(args.dev_ref_path_list)
    print("test_ref_path_list:")
    print(args.test_ref_path_list)
    
    # Set seed
    set_seed(args.seed)

    tokenizer = tokenization.BasicTokenizer(
        os.path.join(args.data_dir, args.vocab_file_name),
        lower_case=args.lower_case,
    )
    args.vocab_size = tokenizer.get_vocab_size()
    args.pad_id = tokenizer.PAD_ID
    args.sos_id = tokenizer.SOS_ID
    args.eos_id = tokenizer.EOS_ID
    args.unk_id = tokenizer.UNK_ID

    if args.emb_path:
        embedding = load_embedding(args.emb_path, tokenizer.index2token)
    else:
        embedding = None
    model = get_model(args, embedding)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    save_config(args)
    save_tokenizer(args, tokenizer)

    bleu_evaluator = BLEUEvaluator()

    if args.mode == "train":
        train_data = trainers.load_and_cache_data(args, data_name="train", tokenizer=tokenizer)
        dev_data = trainers.load_and_cache_data(args, data_name="dev", tokenizer=tokenizer)
        
        trainer = get_trainer(
            args=args,
            model=model,
            train_data=train_data,
            dev_data=dev_data,
            tokenizer=tokenizer,
        )
        args = trainer.args
        save_config(args)

        trainer.train()
        
        return

    else:
        if args.mode == "dev":
            eval_data = trainers.load_and_cache_data(args, data_name="test", tokenizer=tokenizer)
        elif args.mode == "test":
            eval_data = trainers.load_and_cache_data(args, data_name="test", tokenizer=tokenizer)
        else:
            raise ValueError("mode (%s) is invalid" % args.mode)

        trainer = get_trainer(
            args=args,
            model=model,
            test_data=eval_data,
            tokenizer=tokenizer,
        )
        args = trainer.args

        eval_bleu = trainer.test()



def get_output_dir(args):

    # emb_size_str = "emb_size_{}".format(args.emb_size)
    bi_str = "_bi" if args.enc_bidirectional else ""
    lower_case_str = "_lower_case" if args.lower_case else ""
    shuffle_str = "_shuffle" if args.shuffle else ""

    new_output_dir = os.path.join(
        args.output_dir,
        f"{args.model_name}"\
            f"{bi_str}"\
            f"_{args.cell}"\
            f"_hs_{args.hidden_size}"\
            f"_nl_{args.enc_num_layers}"\
            f"_emb_{args.emb_size}"\
            f"_lr_{args.lr}"\
            f"_bs_{args.batch_size}"\
            f"_nep_{args.num_train_epochs}"\
            f"_dep_{args.decay_epoch}"\
            f"_max_gn_{args.max_grad_norm}"\
            f"_init_{args.init}"\
            f"_dp_{args.dropout}"\
            f"{lower_case_str}"\
            f"{shuffle_str}"
    )
    return new_output_dir


if __name__ == "__main__":
    args = load_arguments()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.mode == "train":
        args.output_dir = get_output_dir(args)
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    print("args.output_dir is", args.output_dir)
    log_file = os.path.join(args.output_dir, "stdout_%s.out" % args.mode)
    print("Logging to {} besides stdout".format(log_file))
    sys.stdout = misc.Logger(log_file)

    main(args)
