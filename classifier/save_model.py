import os
import sys
import time
import argparse
from scipy.io import loadmat
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
from classifier.classifier_args import *
import classifier.models as models
from utils import tokenization, optimization, constants, misc
from utils.data import ClassifierDatasetWithoutLab, ClassifierPaddingCollateWithoutLab

def load_arguments():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    add_classifier_arguments(parser)
    add_cnn_classifier_arguments(parser)
    add_rnn_classifier_arguments(parser)
    args = parser.parse_args()

    return args
    
def get_model(args):
    model = getattr(models, args.model_name)(
        config=args,
    )
    
    model.load_state_dict(
        torch.load(
            os.path.join(args.model_path, "model_state_dict.pt")
        )
    )

    param_count = 0
    for param in model.parameters():  # param is a tensor
        param_count += param.view(-1).size()[0]
    
    print("")
    print(repr(model))
    print('Total number of parameters: %d\n' % param_count)
    
    return model

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
    
    tokenizer = tokenization.BasicTokenizer(
        os.path.join(args.data_dir, args.vocab_file_name),
        lower_case=args.lower_case,
    )
    args.vocab_size = tokenizer.get_vocab_size()
    args.pad_id = tokenizer.PAD_ID
    model = get_model(args)
    
    save_dir = args.output_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(model, os.path.join(save_dir, f"classifier.pt"))
    torch.save(tokenizer, os.path.join(save_dir, f"tokenizer.pt"))


if __name__ == "__main__":
    args = load_arguments()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print("args.output_dir is", args.output_dir)
    log_file = os.path.join(args.output_dir, "stdout_%s.out" % args.mode)
    print("Logging to {} besides stdout".format(log_file))
    sys.stdout = misc.Logger(log_file)

    main(args)
