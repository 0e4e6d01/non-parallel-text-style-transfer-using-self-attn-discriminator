import argparse


def add_common_arguments(parser):
    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="input data dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="the output directory where the model predictions and checkpoints will be written",
    )

    # NLP arguments
    parser.add_argument(
        "--vocab_file_name",
        default='vocab',
        type=str,
        help="name of vocab file",
    )
    parser.add_argument(
        "--emb_size",
        type=int,
        default=300,
        help="size of embedding vector"
    )
    parser.add_argument(
        "--lower_case",
        action="store_true",
        help="whether to lower case",
    )
    parser.add_argument(
        "--use_sos",
        action="store_true",
        help="use sos token or not"
    )
    parser.add_argument(
        "--use_eos",
        action="store_true",
        help="use eos token or not"
    )

    # Other arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate"
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=1e-8,
        help="eps for adam"
    )
    parser.add_argument(
        "--l2_reg",
        type=float,
        default=0.0,
        help="L2 regularization"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="number of training epoches"
    )
    parser.add_argument(
        "--decay_epoch",
        type=int,
        default=-1,
        help="epoch when decay starts"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="whether having the data reshuffled at every epoch"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=500,
        help="print log per log_interval step"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=100,
        help="eval per eval_interval step"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="max gradient"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout rate"
    )
    parser.add_argument(
        "--freeze_embedding",
        action="store_true",
        help="freeze embedding during training for pre-trained embedding",
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        help="method for initialization:"+\
            "xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="method for optimization:"+\
            "sgd, adagrad, adadelta, adam"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="path to pre-trained model",
    )
    parser.add_argument(
        "--emb_path",
        default=None,
        type=str,
        help="path to pre-trained embedding",
    )
    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        help="train, dev, train"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="avoid using CUDA when available"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="how many subprocesses to use for data loading"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--grad_accum_interval",
        type=int,
        default=1,
        help="number of steps for gradient accumulation"
    )

    parser.add_argument(
        "--freeze_emb_at_beginning",
        action="store_true",
        help="whether to freeze emb at beginning"
    )
    parser.add_argument(
        "--unfreeze_at_ep",
        type=int,
        default=1,
        help="unfreeze emb at which epoch"
    )