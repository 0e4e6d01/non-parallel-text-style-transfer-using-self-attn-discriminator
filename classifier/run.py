import os
import sys
import time
import argparse
import yaml
import csv

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data.dataloader import DataLoader
# import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.append(".")
# sys.path.append("..")
from common_args import add_common_arguments
from classifier.classifier_args import *
import classifier.models as models
import classifier.trainers as trainers
from utils import tokenization, optimization, constants, misc

def load_arguments():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    add_classifier_arguments(parser)
    add_cnn_classifier_arguments(parser)
    add_rnn_classifier_arguments(parser)
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

def load_embedding(path, index2token):
    print("load embedding")
    emb_dict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            splited = line.strip().split()
            token = splited[0]
            # print(len(splited))
            emb_dict[token] = list(map(float, splited[1:]))

    print(f"len(emb_dict) is {len(emb_dict)}")
    print(f"len(index2token) is {len(index2token)}")
    assert len(emb_dict) == len(index2token)
    embedding = []
    for token in index2token:
        embedding.append(emb_dict[token])
    
    return torch.FloatTensor(embedding)
        

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
    
def get_model(args, embedding=None):
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
    
    # Set seed
    set_seed(args.seed)

    tokenizer = tokenization.BasicTokenizer(
        os.path.join(args.data_dir, args.vocab_file_name),
        lower_case=args.lower_case,
    )
    args.vocab_size = tokenizer.get_vocab_size()
    args.pad_id = tokenizer.PAD_ID

    if args.emb_path:
        embedding = load_embedding(args.emb_path, tokenizer.index2token)
    else:
        embedding = None

    model = get_model(args, embedding)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    save_config(args)
    save_tokenizer(args, tokenizer)

    if args.mode == "train":
        train_data = trainers.load_and_cache_data(args, data_name="train", tokenizer=tokenizer)
        eval_data = trainers.load_and_cache_data(args, data_name="dev", tokenizer=tokenizer)
        
        trainer = get_trainer(
            args=args,
            model=model,
            train_data=train_data,
            dev_data=eval_data,
            tokenizer=tokenizer,
        )
        args = trainer.args
        save_config(args)

        best_acc, train_record, eval_record = trainer.train()
        
        return best_acc

    elif args.mode == "dev" or args.mode == "test":
        if args.mode == "dev":
            eval_data = trainers.load_and_cache_data(args, data_name="dev", tokenizer=tokenizer)
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
        acc = trainer.test()

        print("acc: %.4f" % acc)
        with open(os.path.join(args.output_dir, "evaluating_result.log"), 'w') as f:
            f.write("acc: %.4f" % acc)

    else:
        raise ValueError("mode (%s) is invalid" % args.mode)


def get_output_dir(args):

    # emb_size_str = "emb_size_{}".format(args.emb_size)
    lower_case_str = "_lower_case" if args.lower_case else ""
    shuffle_str = "_shuffle" if args.shuffle else ""

    new_output_dir = os.path.join(
        args.output_dir,
        f"{args.model_name}"\
            f"_emb_size_{args.emb_size}"\
            f"_lr_{args.lr}"\
            f"_bs_{args.batch_size}"\
            f"_nep_{args.num_train_epochs}"\
            f"_dep_{args.decay_epoch}"\
            f"_max_gn_{args.max_grad_norm}"\
            f"_init_{args.init}"\
            f"_dropout_{args.dropout}"\
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
