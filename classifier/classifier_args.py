import argparse

def add_classifier_arguments(parser):
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="number of classes of labels"
    )



def add_cnn_classifier_arguments(parser):
    parser.add_argument(
        "--num_kernels_each_size",
        type=int,
        default=2,
        help="number of kernels of each size"
    )
    parser.add_argument(
        "--kernel_sizes",
        type=int,
        default=[2, 3, 4],
        nargs='+',
        help="a list of kernel sizes"
    )
    

def add_rnn_classifier_arguments(parser):
    parser.add_argument(
        "--enc_num_layers",
        type=int,
        default=1,
        help="number of layer of encoder rnn"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="hidden size of rnn encoder"
    )
    parser.add_argument(
        "--cell",
        type=str,
        default="LSTM",
        help="lstm or gru"
    )
    parser.add_argument(
        "--enc_bidirectional",
        action="store_true",
        help="whether encoder is bi-directional"
    )
