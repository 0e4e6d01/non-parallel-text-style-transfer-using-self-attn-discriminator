import argparse

def add_transfer_arguments(parser):
    # parser.add_argument(
    #     "--ref_dir",
    #     type=str,
    #     required=True,
    #     help="reference dir",
    # )
    parser.add_argument(
        "--dev_ref_dir",
        type=str,
        help="dev reference dir",
    )
    parser.add_argument(
        "--test_ref_dir",
        type=str,
        help="test reference dir",
    )

    parser.add_argument(
        "--style_size",
        type=int,
        default=32,
        help="size of style vector"
    )
    parser.add_argument(
        "--content_size",
        type=int,
        default=512,
        help="size of content vector"
    )
    parser.add_argument(
        "--max_decoding_len",
        type=int,
        default=512,
        help="max length of decoding text"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="beam size for beam search"
    )
    parser.add_argument(
        "--LM_0_path",
        type=str,
        default="lm/pretrained/lm_0.pt",
        help="reference dir",
    )
    parser.add_argument(
        "--LM_1_path",
        type=str,
        default="lm/pretrained/lm_1.pt",
        help="reference dir",
    )

    # noise arguments
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.0,
        help="prob for masking a token"
    )
    parser.add_argument(
        "--drop_prob",
        type=float,
        default=0.0,
        help="prob for droping a token"
    )
    parser.add_argument(
        "--shuffle_weight",
        type=float,
        default=0.0,
        help="weight for shuffle, works only when it is greater than 1"
    )

    parser.add_argument(
        "--cls_model_path",
        type=str,
        default=None,
        help="classifier path",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="number of classes for classifier",
    )

    parser.add_argument(
        "--cnn_clf_path",
        type=str,
        default=None,
        help="cnn classifier path for transfer acc.",
    )

    parser.add_argument(
        "--use_bpe",
        action="store_true",
        help="whether use bpe"
    )

    # some weights
    parser.add_argument(
        "--cls_weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ca_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--bt_weight",
        type=float,
        default=1.0,
    )

def add_rnn_arguments(parser):

    # RNN arguments
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="hidden size of rnn encoder"
    )
    parser.add_argument(
        "--cell",
        type=str,
        default="LSTM",
        help="LSTM or GRU"
    )
    parser.add_argument(
        "--enc_bidirectional",
        action="store_true",
        help="whether encoder is bi-directional"
    )
    parser.add_argument(
        "--enc_num_layers",
        type=int,
        default=1,
        help="number of layer of encoder rnn"
    )
    parser.add_argument(
        "--dec_num_layers",
        type=int,
        default=1,
        help="number of layer of decoder rnn"
    )


def add_cnn_arguments(parser):
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
    
if __name__ == "__main__":
    import os
    import yaml
    def save_config(args):
        # print("args.in_channels is", args.in_channels)
        print("args.enc_num_layers is", args.enc_num_layers)
        print("args.hidden_size is", args.hidden_size)
        print("args.cell is", args.cell)
        print("args.enc_bidirectional is", args.enc_bidirectional)

        with open("test_lm_args.yaml", 'w') as f:
            yaml.dump(args, f)

    parser = argparse.ArgumentParser()
    add_rnn_lm_arguments(parser)
    args = parser.parse_args()

    save_config(args)

    args.test = 1
    print(args.test)
