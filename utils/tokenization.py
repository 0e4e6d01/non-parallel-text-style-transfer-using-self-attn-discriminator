import os
import sys
sys.path.append(".")
sys.path.append("..")

from utils import constants


def read_vocab_file(vocab_file):
    token2index = {}
    index2token = []
    with open(vocab_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            token = line.strip()
            if token != '':
                token2index[token] = len(index2token)
                index2token.append(token)
    return token2index, index2token

class SimpleVocab:
    
    def __init__(self, vocab_file, **kwargs):
        self.PAD_TOKEN = kwargs.pop("pad_token", constants.PAD_TOKEN)
        self.UNK_TOKEN = kwargs.pop("unk_token", constants.UNK_TOKEN)
        self.SOS_TOKEN = kwargs.pop("sos_token", constants.SOS_TOKEN)
        self.EOS_TOKEN = kwargs.pop("eos_token", constants.EOS_TOKEN)
        self.token2index, self.index2token = read_vocab_file(vocab_file)

        keywords = [
            ("PAD_ID", self.PAD_TOKEN),
            ("UNK_ID", self.UNK_TOKEN),
            ("SOS_ID", self.SOS_TOKEN),
            ("EOS_ID", self.EOS_TOKEN),
        ]
        for i, t in keywords:
            try:
                setattr(self, i, self.token2index[t])
                print(self.token2index[t], i)
            except KeyError:
                self.token2index[t] = len(self.index2token)
                self.index2token.append(t)
                setattr(self, i, self.token2index[t])
                print(self.token2index[t], i)

        # self.UNK_ID = self.token2index[self.UNK_TOKEN]
        # self.SOS_ID = self.token2index[self.SOS_TOKEN]
        # self.EOS_ID = self.token2index[self.EOS_TOKEN]

    def token_to_index(self, token_list):
        index_list = []
        for token in token_list:
            index_list.append(self.token2index.get(token, self.UNK_ID))
        return index_list
    
    def index_to_token(self, index_list, include_sos_eos=True):
        token_list = []
        for index in index_list:
            if include_sos_eos or (index != self.SOS_ID and index != self.EOS_ID):
                token = self.index2token[index]
                token_list.append(token)
                
            if index == self.EOS_ID:
                break
                
        return token_list
    
    def get_vocab_size(self):
        return len(self.index2token)

class BasicTokenizer(SimpleVocab):

    def __init__(self, vocab_file, lower_case=True, pad_token=constants.PAD_TOKEN,
                unk_token=constants.UNK_TOKEN, sos_token=constants.SOS_TOKEN,
                eos_token=constants.EOS_TOKEN):
        super(BasicTokenizer, self).__init__(
            vocab_file, pad_token=constants.PAD_TOKEN, unk_token=constants.UNK_TOKEN,
            sos_token=constants.SOS_TOKEN, eos_token=constants.EOS_TOKEN
        )
        self.lower_case = lower_case

    def tokenize(self, text):
        _token_list = text.strip().split()
        token_list = []
        for token in _token_list:
            if self.lower_case:
                token = token.lower()
            token_list.append(token)
        return token_list
    
    def encode(self, text):
        """
        encode text into index list
        """
        token_list = self.tokenize(text)
        index_list = self.token_to_index(token_list)
        return index_list
    
    def decode(self, index_list, include_sos_eos=True):
        """
        decode index list into text
        """
        token_list = self.index_to_token(index_list, include_sos_eos)
        text = ' '.join(token_list)
        return text

