import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class CNNClassifier(nn.Module):
    
    def __init__(self, config, embedding=None):
        super(CNNClassifier, self).__init__()

        pad_id = getattr(config, "pad_id", 0)
        dropout = getattr(config, "dropout", 0.0)
        self.num_kernels_each_size = config.num_kernels_each_size
        self.kernel_sizes = config.kernel_sizes

        # print('embedding.shape is', embedding.shape)
        in_channels = 1  # only use one embedding
        hidden_size = config.num_kernels_each_size*len(config.kernel_sizes)
        
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding),
            freeze=config.freeze_embedding,
            padding_idx=pad_id,
        ) if embedding is not None else \
            nn.Embedding(config.vocab_size, config.emb_size, padding_idx=pad_id)
        
        # input is a tensor of [batch_size, n_emb, max_len, emb_size], i.e. [batch_size, in_channels, H, W]
        self.cnns = nn.ModuleList([nn.Conv2d(in_channels, config.num_kernels_each_size,
                                (ks, config.emb_size)) for ks in config.kernel_sizes])

        self.ffn = nn.Linear(hidden_size, config.num_labels)

        self.dropout_layer = nn.Dropout(dropout)
        
        self.device = config.device
        #self.softmax = nn.log_softmax()
        
        # if self.weight is not None:
        #     self.criterion = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        # else:
        #     self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, src, soft=False):
        """
        Args:
            src: shape is [batch_size, max_src_len]
        Returns:
            prob: shape is [batch_size, num_labels]
        """
        # Convert word indexes to embeddings
        if soft:
            embedded = src @ self.embedding.weight
        else:
            embedded = self.embedding(src)
        # shape of embedded is [batch_size, max_len, emb_size]
        embedded = self.dropout_layer(embedded)

        outputs = []
        for i, cnn in enumerate(self.cnns):
            if self.kernel_sizes[i] <= embedded.shape[1]:
                # result of cnn is [batch_size, num_kernels_each_size, f(max_len, kernel_size), 1]
                temp = cnn(embedded.unsqueeze(1)).squeeze(3)
                temp = F.relu(temp)
                temp = F.max_pool1d(temp, temp.size(2)).squeeze(2)
                # print("temp.shape is", temp.shape)  # torch.Size([128, 2])
            else:
                temp = torch.zeros(embedded.shape[0], self.num_kernels_each_size, device=self.device)
            outputs.append(temp)
        outputs = torch.cat(outputs, dim=1)
        # shape of outputs is [batch_size, hidden_size]
        return self.ffn(outputs)


class SelfAttnRNNClassifier(nn.Module):
    """
    Self Attn RNN Classifier
    """
    def __init__(self, config, embedding=None):
        super(SelfAttnRNNClassifier, self).__init__()

        self.cell = getattr(config, "cell", "LSTM").upper()
        self.bidirectional = getattr(config, "enc_bidirectional", True)
        pad_id = getattr(config, "pad_id", 0)
        dropout = getattr(config, "dropout", 0.0)
        num_layers = getattr(config, "enc_num_layers", 1)

        self.mask_id = getattr(config, "mask_id", None)

        self.hidden_size = config.hidden_size
        # attention_size = getattr(config, "attention_size", self.hidden_size)

        print("[SelfAttnRNNClassifier.__init__] type(embedding) is", type(embedding))
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding),
            freeze=config.freeze_embedding,
            padding_idx=pad_id
        ) if embedding is not None else \
            nn.Embedding(config.vocab_size, config.emb_size, padding_idx=pad_id)

        assert self.cell == "LSTM" or self.cell == "GRU"
        # Initialize RNN
        self.rnn = getattr(nn, self.cell)(
            input_size=config.emb_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=(0.0 if num_layers == 1 else dropout),
            bidirectional=self.bidirectional,
        )

        num_hidden = 2 if self.cell == "LSTM" else 1
        self.output_bridge = nn.Linear(
            in_features=2*self.hidden_size if self.bidirectional\
                else self.hidden_size,
            out_features= self.hidden_size
        )
        self.hidden_bridges = nn.ModuleList([
            nn.Linear(
                in_features=2*self.hidden_size if self.bidirectional\
                    else self.hidden_size,
                out_features=self.hidden_size
            ) for _ in range(num_hidden)
        ])

        self.dropout_layer = nn.Dropout(dropout)

        self.linear_emb = nn.Linear(
            in_features=config.emb_size,
            out_features=self.hidden_size,
        )

        self.linear_q = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
        )
        self.linear_attn = nn.Linear(
            in_features=self.hidden_size,
            out_features=1,
        )
        self.output_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=config.num_labels,
        )

        self.device = config.device

    def set_mask_id(self, mask_id):
        self.mask_id = mask_id

    def freeze_emb(self):
        print("freeze all param in embedding")
        for param in self.embedding.parameters():
            param.requires_grad = False
    
    def unfreeze_emb(self):
        print("unfreeze params in classifier")
        for param in self.embedding.parameters():
            param.requires_grad = True

    def attention(self, q, k, v, pad_mask=None):
        """
        Args:
            q: shape is [batch_size, q_size]
            k: shape is [batch_size, max_len, q_size]
            v: shape is [batch_size, max_len, v_size]
            pad_mask: shape is [batch_size, max_len], 1 for not padding and 0 for padding
        Returns:
            context: shape is [batch_size, v_size]
            weights: shape is [batch_size, max_len]
        """
        assert q.shape[1] == k.shape[2]
        assert k.shape[1] == v.shape[1]

        mid = torch.tanh(k + self.linear_q(q).unsqueeze(1))
        # mid = torch.tanh(self.linear_k(k) + self.linear_q(q).unsqueeze(1))
        # shape of mid is [batch_size, max_len, q_size]
        logits = self.linear_attn(mid).squeeze(2)
        # logits = torch.sum(mid, dim=2)
        # shape of logits is [batch_size, max_len]


        if pad_mask is not None:
            # mask out [sos] [eos] tokens
            pad_mask = pad_mask[:, 2:]
            tmp = torch.ones((pad_mask.shape[0],1), dtype=bool).to(self.device)
            pad_mask = torch.cat([tmp, pad_mask, tmp], dim=1)


            logits[pad_mask] = float('-inf')
            weights = F.softmax(logits, dim=1)
        else:
            weights = F.softmax(logits, dim=1)

        context = torch.bmm(weights.unsqueeze(1), v).squeeze(1)
        # shape of context is [batch_size, v_size]
        return context, weights

        
    def forward(self, src, src_len, pad_mask=None, hidden=None, soft=False):
        """
        Args:
            src: shape is [batch_size, max_src_len]
            src_len: shape is [batch_size]
            pad_mask: shape is [batch_size, max_len], 1 for padding and 0 for not padding
        Returns:
            logits: shape is [batch_size, num_label]
            weights: shape is [batch_size, max_len]
        """
        src = src.transpose(0,1)
        # Convert word indexes to embeddings
        if soft:
            embedded = src @ self.embedding.weight
        else:
            embedded = self.embedding(src)
        # shape of embedded is [max_src_len, batch_size, embedding_size]
        embedded = self.dropout_layer(embedded)
        # Pack padded batch of sequences for RNN module
        packed = pack(embedded, src_len)
        # Forward pass through RNN
        outputs, hidden = self.rnn(packed, hidden)
        # Unpack padding
        outputs, _ = unpack(outputs)
        # shape of outputs is [max_src_len, batch_size, num_direction*hidden_size]
        # for LSTM: hidden is a tuple (h_n, c_n)
        # h_n has shape of [num_layers*num_directions, batch_size, hidden_size]
        # c_n has shape of [num_layers*num_directions, batch_size, hidden_size]
        # for GRU: hidden is a tensor h_n
        # h_n has shape of [num_layers*num_directions, batch_size, hidden_size]

        outputs = self.output_bridge(outputs)
        # shape of outputs is [max_src_len, batch_size, hidden_size]
        outputs = outputs.transpose(0,1)
        # shape of outputs is [batch_size, max_src_len, hidden_size]
        
        batch_size = outputs.shape[0]
        num_directions = 2 if self.bidirectional else 1


        k = self.linear_emb(embedded).transpose(0,1)
        # shape of k is [batch_size, max_src_len, hidden_size]

        if self.cell == "LSTM":
            hidden = (
                hidden[0].view(-1, num_directions, batch_size, self.hidden_size)[-1],
                hidden[1].view(-1, num_directions, batch_size, self.hidden_size)[-1],
            )
            hidden = (
                self.hidden_bridges[0](hidden[0].transpose(0,1).reshape(-1 ,num_directions*self.hidden_size)),
                self.hidden_bridges[1](hidden[1].transpose(0,1).reshape(-1 ,num_directions*self.hidden_size)),
            )
            context, weights = self.attention(
                hidden[0], k, k, pad_mask,
            )
        else:
            hidden = hidden.view(-1, num_directions, batch_size, self.hidden_size)[-1]
            hidden = self.hidden_bridges[0](hidden.transpose(0,1).reshape(-1 ,num_directions*self.hidden_size))
            context, weights = self.attention(
                hidden, k, k, pad_mask,
            )
        
        logits = self.output_layer(context)
        # shape of logits is [batch_size, num_label]
        # print("[SelfAttnRNNClassifier.forward] logits.shape is", logits.shape)
        # Return output and final hidden state
        return logits, weights


    def get_masked_src(self, src, src_len, pad_mask):
        """
        Args:
            src: shape is [batch_size, max_src_len]
            src_len: shape is [batch_size]
            lab: shape is [batch_size]
            pad_mask: shape is [batch_size, max_len], 1 for padding and 0 for not padding
        """
        if self.mask_id is None:
            print("[SelfAttnRNNClassifier.get_masked_src] set mask id to 0")
            self.mask_id = 0

        _pad_mask = pad_mask[:, 2:]
        tmp = torch.ones((pad_mask.shape[0],1), dtype=bool).to(self.device)
        pad_sos_eos_mask = torch.cat([tmp, _pad_mask, tmp], dim=1)

        cls_logits, style_alpha = self.forward(src, src_len, pad_mask)
        # shape of cls_logits is [batch_size, num_label]
        # shape of style_alpha is [batch_size, max_src_len]

        threshold = (1.0 / (src_len-2).float()).unsqueeze(1)

        inv_pad_sos_eos_mask = torch.logical_not(pad_sos_eos_mask)
        style_mask = (style_alpha <= threshold) * inv_pad_sos_eos_mask
        content_mask = (style_alpha > threshold) * inv_pad_sos_eos_mask
        style_src = src.clone()
        content_src = src.clone()
        style_src[style_mask] = self.mask_id
        content_src[content_mask] = self.mask_id

        return cls_logits, style_alpha, style_src, content_src
