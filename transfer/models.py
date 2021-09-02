import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class RNNEncoder(nn.Module):
    """
    RNN Encoder
    """
    def __init__(self, config, embedding=None):
        super(RNNEncoder, self).__init__()

        self.cell = getattr(config, "cell", "LSTM").upper()
        self.bidirectional = getattr(config, "enc_bidirectional", True)
        pad_id = getattr(config, "pad_id", 0)
        dropout = getattr(config, "dropout", 0.0)
        num_layers = getattr(config, "enc_num_layers", 1)

        self.hidden_size = config.hidden_size

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
                out_features= self.hidden_size
            ) for _ in range(num_hidden)
        ])

        self.dropout_layer = nn.Dropout(dropout)

        
    def forward(self, src, src_len, hidden=None):
        """
        Args:
            src: shape is [max_src_len, batch_size]
            src_len: shape is [batch_size]
        Returns:
            outputs: shape is [max_src_len, batch_size, hidden_size]
            hidden: For LSTM: (h_n, c_n), shape of h_n is [batch_size, hidden_size], same as c_n
                    For GRU: h_n, shape is [batch_size, hidden_size]
        """
        # Convert word indexes to embeddings
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
        
        batch_size = outputs.shape[1]
        num_directions = 2 if self.bidirectional else 1

        if self.cell == "LSTM":
            hidden = (
                hidden[0].view(-1, num_directions, batch_size, self.hidden_size)[-1],
                hidden[1].view(-1, num_directions, batch_size, self.hidden_size)[-1],
            )
            hidden = (
                self.hidden_bridges[0](hidden[0].transpose(0,1).reshape(-1 ,num_directions*self.hidden_size)),
                self.hidden_bridges[1](hidden[1].transpose(0,1).reshape(-1 ,num_directions*self.hidden_size)),
            )
        else:
            hidden = hidden.view(-1, num_directions, batch_size, self.hidden_size)[-1]
            hidden = self.hidden_bridges[0](hidden.transpose(0,1).reshape(-1 ,num_directions*self.hidden_size))

        # Return output and final hidden state
        return outputs, hidden

class AttnRNNDecoder(nn.Module):
    """
    RNN Decoder with Attn.
    """
    def __init__(self, config=None, embedding=None, style_embedding=None):
        super().__init__()
        if not config:
            return

        cell = getattr(config, "cell", "LSTM").upper()
        self.pad_id = getattr(config, "pad_id", 0)
        dropout = getattr(config, "dropout", 0.0)
        num_layers = getattr(config, "dec_num_layers", 1)
        self.vocab_size = config.vocab_size

        self.beam_size = getattr(config, "beam_size", 5)
        print("self.beam_size is", self.beam_size)

        assert hasattr(config, "sos_id")
        assert hasattr(config, "eos_id")
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id

        self.hidden_size = config.hidden_size
        self.max_decoding_len = config.max_decoding_len

        self.device = config.device

        self.embedding = embedding if embedding is not None else \
            nn.Embedding(self.vocab_size, config.emb_size, padding_idx=self.pad_id)
        
        self.style_embedding = style_embedding if style_embedding is not None\
            else nn.Embedding(
                2,
                config.emb_size,  # self.hidden_size,
            )

        assert cell == "LSTM" or cell == "GRU"
        # Initialize RNN
        self.rnn = getattr(nn, cell)(
            input_size=2*config.emb_size+self.hidden_size,  # word emb, style emb, last_ctx
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=(0.0 if num_layers == 1 else dropout),
            bidirectional=False,
        )

        self.dropout_layer = nn.Dropout(dropout)


        # attention
        self.attention = Attention(self.hidden_size)

        self.ffn_out = nn.Linear(
            in_features=2*self.hidden_size,  # rnn_out, content_ctx
            out_features=self.vocab_size,
        )


    def freeze_emb(self):
        print("[AttnRNNDecoder.freeze_emb] freeze all param in embedding")
        for param in self.embedding.parameters():
            param.requires_grad = False

    def unfreeze_emb(self):
        print("[AttnRNNDecoder.freeze_emb] unfreeze all param in embedding")
        for param in self.embedding.parameters():
            param.requires_grad = True
    
    def set_style_emb(self, lab, inv_lab):
        self.style_emb = self.style_embedding(lab).unsqueeze(0)
        self.inv_style_emb = self.style_embedding(inv_lab).unsqueeze(0)
        # shape of style_emb is [1, batch_size, hidden_size]

    def drop_style_emb(self):
        self.style_emb = None
        self.inv_style_emb = None

    def forward(self, trg, trg_len, lab, memory,
            pad_mask, hidden=None, transfer=False):
        """
        Args:
            trg: shape is [max_trg_len, batch_size] ([SOS] xx xx ... xx [EOS])
            trg_len: shape is [batch_size]
            lab: shape is [batch_size]
            content_memory: shape is [max_src_len, batch_size, hidden_size] ([SOS] xx xx ... xx [EOS])
            style_memory: shape is [max_src_len, batch_size, hidden_size] ([SOS] xx xx ... xx [EOS])
            pad_mask: shape is [batch_size, max_len], 0 for not padding and 1 for padding
            pad_sos_eos_mask: shape is [batch_size, max_len], 0 for not padding and 1 for padding, sos and eos
            hidden: For LSTM: (h_n, c_n), shape of h_n is [num_layers*num_directions, batch_size, hidden_size], same as c_n
                    For GRU: h_n, shape is [num_layers*num_directions, batch_size, hidden_size]
        Returns:
            logits_result: shape is [max_trg_len-1, batch_size, vocab_size]
            align_result: list of tuple of [batch_size, max_content_len] and [batch_size, max_style_len]
        """
        # print("[AttnRNNDecoder.forward]trg is", trg)
        trg = trg[:-1]
        # do not use trg_len -= 1 since it is an in-place op.
        trg_len = trg_len - 1
        max_trg_len, batch_size = trg.shape
        
        assert torch.max(trg_len) == max_trg_len

        if transfer:
            style_emb = self.inv_style_emb
        else:
            style_emb = self.style_emb

        # shape of style_emb is [1, batch_size, hidden_size]
        memory = memory.transpose(0,1)

        last_ctx = torch.zeros((batch_size, memory.shape[-1]),
            requires_grad=False, device=self.device)
            
        logit_list = []
        align_list = []
        for i in range(max_trg_len):
            rnn_in = trg[i]
            rnn_in = rnn_in.unsqueeze(0)
            logits, hidden, align, last_ctx = self.step(rnn_in, lab, memory, last_ctx.unsqueeze(0),
                style_emb, pad_mask, hidden=hidden, soft=False, transfer=transfer)
            # shape of logits is [1, batch_size, vocab_size]
            # align is a tuple of [batch_size, max_content_len] and [batch_size, max_style_len]
            logit_list.append(logits)
            align_list.append(align)

        return torch.cat(logit_list, dim=0), align_list
    
    def step(self, src, lab, memory, last_ctx,
            style_emb, pad_mask, hidden=None, soft=False,
            transfer=False):
        """
        Args:
            src: shape is [1, batch_size] if soft is False, and [1, batch_size, vocab_size] if soft is True
            lab: shape is [batch_size]
            content_memory: shape is [batch_size, max_src_len, hidden_size] ([SOS] xx xx ... xx [EOS])
            style_memory: shape is [batch_size, max_src_len, hidden_size] ([SOS] xx xx ... xx [EOS])
            style_emb: shape is [1, batch_size, hidden_size]

            hidden: For LSTM: (h_n, c_n), shape of h_n is [num_layers*num_directions, batch_size, hidden_size], same as c_n
                    For GRU: h_n, shape is [num_layers*num_directions, batch_size, hidden_size]
        Returns:
            output: shape is [1, batch_size, vocab_size]
            hidden: For LSTM: (h_n, c_n), shape of h_n is [num_layers*num_directions, batch_size, hidden_size], same as c_n
                    For GRU: h_n, shape is [num_layers*num_directions, batch_size, hidden_size]
            weights: tuple of [batch_size, max_content_len] and [batch_size, max_style_len]
        """
        if soft:
            embedded = src @ self.embedding.weight
        else:
            embedded = self.embedding(src)
        # embedded = self.dropout_layer(embedded)
        rnn_in = torch.cat([embedded, style_emb, last_ctx], dim=2)
        # print("[SoftDualAttnRNNDecoder.step] rnn_in.shape is", rnn_in.shape)  # torch.Size([1, 64, 812])
        rnn_in = self.dropout_layer(rnn_in)
        rnn_out, hidden = self.rnn(rnn_in, hidden)
        ctx, weights = self.attention(rnn_out.squeeze(0),
            memory, memory, pad_mask)
        # shape of ctx is [batch_size, hidden_size]
        # shape of weights is [batch_size, max_content_len]


        ffn_in = torch.cat([rnn_out, ctx.unsqueeze(0)], dim=2)
        output = self.ffn_out(ffn_in)
        return output, hidden, weights, ctx

    def sample_decode(self, lab, memory, pad_mask, hidden=None, greedy=False):
        """
        Args:
            lab: shape is [batch_size], it is the labels of input text
            content_memory: shape is [max_src_len, batch_size, hidden_size] ([SOS] xx xx ... xx [EOS])
            style_memory: shape is [max_src_len, batch_size, hidden_size] ([SOS] xx xx ... xx [EOS])
            pad_mask: shape is [batch_size, max_len], 0 for not padding and 1 for padding
            pad_sos_eos_mask: shape is [batch_size, max_len], 0 for not padding and 1 for padding, sos and eos
            hidden: For LSTM: (h_n, c_n), shape of h_n is [num_layers*num_directions, batch_size, hidden_size], same as c_n
                    For GRU: h_n, shape is [num_layers*num_directions, batch_size, hidden_size]
        Returns:
            outputs_result: shape is [max_len, batch_size]
            output_len: shape is [batch_size]
            logits_result: shape is [max_len, batch_size, vocab_size]
            align_result: list of tuple of [batch_size, max_content_len] and [batch_size, max_style_len]
        """
        # style_emb = self.style_embedding(lab).unsqueeze(0)
        style_emb = self.inv_style_emb
        # shape of style_emb is [1, batch_size, hidden_size]
        memory = memory.transpose(0,1)
        batch_size = lab.shape[0]

        rnn_in = torch.LongTensor([self.sos_id] * batch_size).unsqueeze(0).to(self.device)
        sos_tokens = torch.LongTensor([self.sos_id] * batch_size).to(self.device)
        eos_tokens = torch.LongTensor([self.eos_id] * batch_size).to(self.device)

        last_ctx = torch.zeros((batch_size, memory.shape[-1]),
            requires_grad=False, device=self.device)

        mask = torch.ones_like(eos_tokens, dtype=torch.bool).to(self.device)
        # mask: 1 for still decoding, 0 for stop
        cur_decoding_len = 0
        output_len = torch.zeros_like(mask, dtype=torch.int16)
        output = []
        logit_list = []
        align_list = []

        while torch.sum(mask) != 0 and cur_decoding_len <= self.max_decoding_len:
            # logits, hidden, align = self.step(rnn_in, lab, memory,
            #     style_emb, pad_mask, hidden=hidden, soft=False, transfer=True)
            logits, hidden, align, last_ctx = self.step(rnn_in, lab, memory, last_ctx.unsqueeze(0),
                style_emb, pad_mask, hidden=hidden, soft=False, transfer=True)
            # shape of logits is [1, batch_size, vocab_size]
            logit_list.append(logits)
            align_list.append(align)
            logits = logits.squeeze(0)
            if greedy:
                rnn_in_id = torch.argmax(logits, dim=1)
            else:
                # to be corrected
                probs = F.softmax(logits)
                rnn_in_id = torch.multinomial(probs, num_samples=1)
            cur_decoding_len += 1
            rnn_in = rnn_in_id.unsqueeze(0)
            output.append(rnn_in)
            
            output_len += mask
            mask = torch.mul((rnn_in_id != eos_tokens), mask)
        
        out = torch.cat(output, dim=0)

        # add sos
        out = torch.cat([sos_tokens.unsqueeze(0), out], dim=0)
        output_len += 1

        if cur_decoding_len > self.max_decoding_len:
            add_eos = False
            for ind, m in enumerate(mask):
                # if m==1 and out[ind, -1]!=self.eos_id:
                if m==1 and out[-1, ind]!=self.eos_id:
                    add_eos = True
                    output_len[ind] += 1
            if add_eos:
                out = torch.cat([out, eos_tokens.unsqueeze(0)], dim=0)
        
        # =====================
        # set padding
        tmp = torch.arange(out.shape[0]).unsqueeze(-1).to(self.device).expand(-1,out.shape[1])
        out[tmp>=output_len.unsqueeze(0)] = self.pad_id
        # =====================

        return out, output_len, torch.cat(logit_list, dim=0),\
            align_list

    def beam_decode(self, lab, memory, pad_mask, hidden=None, beam_size=None):
        """
        Args:
            lab: shape is [batch_size], it is the labels of input text
            content_memory: shape is [max_src_len, batch_size, hidden_size] ([SOS] xx xx ... xx [EOS])
            style_memory: shape is [max_src_len, batch_size, hidden_size] ([SOS] xx xx ... xx [EOS])
            pad_mask: shape is [batch_size, max_len], 0 for not padding and 1 for padding
            pad_sos_eos_mask: shape is [batch_size, max_len], 0 for not padding and 1 for padding, sos and eos
            hidden: For LSTM: (h_n, c_n), shape of h_n is [num_layers*num_directions, batch_size, hidden_size], same as c_n
                    For GRU: h_n, shape is [num_layers*num_directions, batch_size, hidden_size]
        Returns:
            outputs_result: shape is [max_len, batch_size]
            output_len: shape is [batch_size]
            logits_result: shape is [max_len, batch_size, vocab_size]
            align_result: list of tuple of [batch_size, max_content_len] and [batch_size, max_style_len]
        """
        if beam_size is None:
            beam_size = self.beam_size
        # style_emb = self.style_embedding(lab).unsqueeze(0)
        all_style_emb = self.inv_style_emb
        # shape of all_style_emb is [1, batch_size, hidden_size]
        all_memory = memory.transpose(0,1)
        all_pad_mask = pad_mask
        all_lab = lab
        batch_size = all_lab.shape[0]

        sos_tokens = torch.LongTensor([self.sos_id] * beam_size).to(self.device)
        eos_tokens = torch.LongTensor([self.eos_id] * beam_size).to(self.device)

        init_hidden = hidden
        is_lstm = isinstance(init_hidden, tuple)
        # print("init_hidden[0].shape is", init_hidden[0].shape)  # torch.Size([1, 32, 512])
        out_hyp_list = []
        for batch_ind in range(batch_size):
            completed_hyp = []
            active_hyp = [
                Hypothesis(
                    out=[self.sos_id],
                    hidden=(init_hidden[0][:, batch_ind].unsqueeze(1),
                        init_hidden[1][:, batch_ind].unsqueeze(1))\
                        if is_lstm else init_hidden[:, batch_ind].unsqueeze(1),
                    last_ctx=torch.zeros((1, memory.shape[-1]),
                        requires_grad=False, device=self.device),
                    score=0.0,
                )
            ]
            style_emb = all_style_emb[:, batch_ind].unsqueeze(1)
            # shape of style_emb is [1, 1, hidden_size]
            lab = all_lab[batch_ind].unsqueeze(0)
            memory = all_memory[batch_ind].unsqueeze(0)
            pad_mask = all_pad_mask[batch_ind].unsqueeze(0)

            cur_decoding_len = 0
            output = []

            while len(completed_hyp) < beam_size and cur_decoding_len <= self.max_decoding_len:
                cur_decoding_len += 1
                new_hyp_score_list = []
                for hyp_ind, hyp in enumerate(active_hyp):
                    rnn_in = torch.LongTensor([hyp.out[-1]]).unsqueeze(0).to(self.device)
                    logits, hyp.hidden, align, hyp.last_ctx = self.step(rnn_in,
                        lab, memory, hyp.last_ctx.unsqueeze(0), style_emb, pad_mask,
                        hidden=hyp.hidden, soft=False, transfer=True)
                    # shape of logits is [1, 1, vocab_size]

                    logits = logits.squeeze(0).squeeze(0)
                    log_probs = F.log_softmax(logits, dim=-1)
                    new_hyp_score = log_probs + hyp.score
                    new_hyp_score_list.append(new_hyp_score)

                num_active_hyp = beam_size - len(completed_hyp)
                new_hyp_score_tensor = torch.stack(new_hyp_score_list, dim=0).reshape(-1)
                topk_v, topk_i = torch.topk(new_hyp_score_tensor, k=num_active_hyp)
                prev_hyp_ind = topk_i // self.vocab_size
                new_token_ind = topk_i % self.vocab_size
                # print("num_active_hyp is", num_active_hyp)
                # print("topk_i is", topk_i)
                # print("prev_hyp_ind is", prev_hyp_ind)
                # print("new_token_ind is", new_token_ind)
                
                new_active_hyp = []
                for i in range(num_active_hyp):
                    prev_hyp = active_hyp[prev_hyp_ind[i]]
                    new_hyp = Hypothesis(
                        out=prev_hyp.out+[new_token_ind[i].item()],
                        hidden=prev_hyp.hidden,
                        last_ctx=prev_hyp.last_ctx,
                        score=topk_v[i].item(),
                    )
                    if new_hyp.out[-1]==self.eos_id:
                        completed_hyp.append(new_hyp)
                    else:
                        new_active_hyp.append(new_hyp)
                active_hyp = new_active_hyp
                
        
            if len(completed_hyp)==0:  # no completed
                completed_hyp.append(active_hyp[0])
            
            sorted_hyp = sorted(completed_hyp, key=lambda x: x.score/len(x.out), reverse=True)
            out_hyp_list.append(sorted_hyp[0])

        out_list = []
        out_len_list = []
        for hyp in out_hyp_list:
            out_list.append(hyp.out)
            out_len_list.append(len(hyp.out))
        # print("out_list is")
        # print(out_list)
        out = torch.zeros(max(out_len_list), batch_size, dtype=torch.int64, device=self.device)
        for i, o in enumerate(out_list):
            out[:out_len_list[i], i] = torch.LongTensor(o)

        return out, torch.IntTensor(out_len_list).to(self.device)

    def soft_decode(self, lab, memory, pad_mask, hidden=None):
        """
        Args:
            lab: shape is [batch_size], it is the labels of input text
            content_memory: shape is [max_src_len, batch_size, hidden_size] ([SOS] xx xx ... xx [EOS])
            style_memory: shape is [max_src_len, batch_size, hidden_size] ([SOS] xx xx ... xx [EOS])
            pad_mask: shape is [batch_size, max_len], 0 for not padding and 1 for padding
            pad_sos_eos_mask: shape is [batch_size, max_len], 0 for not padding and 1 for padding, sos and eos
            hidden: For LSTM: (h_n, c_n), shape of h_n is [num_layers*num_directions, batch_size, hidden_size], same as c_n
                    For GRU: h_n, shape is [num_layers*num_directions, batch_size, hidden_size]
        Returns:
            outputs: shape is [max_len, batch_size, vocab_size]
            outputs_len: shape is [batch_size]
            ne: shape is [batch_size]
            align_result: list of tuple of [batch_size, max_content_len] and [batch_size, max_style_len]
        """
        # style_emb = self.style_embedding(lab).unsqueeze(0)
        style_emb = self.inv_style_emb
        # shape of style_emb is [1, batch_size, hidden_size]
        memory = memory.transpose(0,1)
        batch_size = lab.shape[0]

        # rnn_in is (soft) one-hot
        rnn_in = torch.zeros(1, 1, self.vocab_size, dtype=torch.float32).to(self.device)
        
        rnn_in[..., self.sos_id] = 1.0
        rnn_in = rnn_in.expand((-1, batch_size, -1))

        pad_vec = torch.zeros(self.vocab_size, dtype=torch.float32).to(self.device)
        pad_vec[self.pad_id] = 1.0

        eos_tokens = torch.LongTensor([self.eos_id] * batch_size).to(self.device)
        # # eos_ohs is one-hot vector
        # eos_ohs = torch.zeros(1, self.vocab_size, dtype=torch.float32).to(self.device)
        # eos_ohs[:, self.eos_id] = 1.0
        # eos_ohs = eos_ohs.expand((batch_size, -1))

        last_ctx = torch.zeros((batch_size, memory.shape[-1]),
            requires_grad=False, device=self.device)

        mask = torch.ones(batch_size, dtype=torch.bool).to(self.device)
        # mask: 1 for still decoding, 0 for stop
        cur_decoding_len = 0
        outputs_len = torch.zeros_like(mask, dtype=torch.int16)
        output_list = []
        # logit_list = []
        align_list = []
        
        while torch.sum(mask) != 0 and cur_decoding_len <= self.max_decoding_len:
            # logits, hidden, align = self.step(rnn_in, content_memory,
            #     style_memory, style_token_embs_cur, style_emb, content_pad_mask,
            #     style_pad_mask, hidden=hidden, soft=True, transfer=True,
            #     style_token_embs_trans=style_token_embs_trans,
            #     transfer_direction=transfer_direction)
            logits, hidden, align, last_ctx = self.step(rnn_in, lab, memory, last_ctx.unsqueeze(0),
                style_emb, pad_mask,  hidden=hidden, soft=True, transfer=True)
            # shape of logits is [1, batch_size, vocab_size]
            # logit_list.append(logits)
            align_list.append(align)
            logits = logits.squeeze(0)
            rnn_in = F.gumbel_softmax(logits, tau=1.0, hard=True)
            # shape of rnn_in is [batch_size, vocab_size]
            rnn_in_id = torch.argmax(rnn_in, dim=1)
            # shape of rnn_in_id is [batch_size]
            rnn_in = rnn_in.unsqueeze(0)
            # shape of rnn_in is [1, batch_size, vocab_size]

            output_list.append(torch.mul(rnn_in, mask.unsqueeze(1).unsqueeze(0)))
            cur_decoding_len += 1
            
            outputs_len += mask
            mask = torch.mul((rnn_in_id != eos_tokens), mask)
            
        # logits = torch.cat(logit_list, dim=0)
        outputs = torch.cat(output_list, dim=0)
        
        # add sos
        # sos_ohs is one-hot vector
        sos_ohs = torch.zeros(1, 1, self.vocab_size, dtype=torch.float32).to(self.device)
        sos_ohs[..., self.sos_id] = 1.0
        sos_ohs = sos_ohs.expand((-1, batch_size, -1))
        # out_w_sos = torch.cat([sos_ohs, out], dim=0)
        out = torch.cat([sos_ohs, outputs], dim=0)
        outputs_len += 1
        
        if cur_decoding_len > self.max_decoding_len:
            add_eos = False
            for ind, m in enumerate(mask):
                if m==1 and torch.argmax(out[-1, ind], dim=-1)!=self.eos_id:
                    add_eos = True
                    outputs_len[ind] += 1
            if add_eos:
                eos_vec = torch.zeros(self.vocab_size, dtype=torch.float32).to(self.device)
                eos_vec[self.eos_id] = 1.0
                eos_vec = eos_vec.unsqueeze(0).unsqueeze(0).expand(-1,out.shape[1],-1)
                out = torch.cat([out, eos_vec], dim=0)
        
        # =====================
        # set padding
        tmp = torch.arange(out.shape[0]).unsqueeze(-1).to(self.device).expand(-1,out.shape[1])
        out[(tmp>=outputs_len.unsqueeze(0))] = pad_vec
        # =====================
        
        return out, outputs_len, align_list
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear_q = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
        )
        # self.linear_k = nn.Linear(
        #     in_features=self.hidden_size,
        #     out_features=self.hidden_size,
        # )
        self.linear_attn = nn.Linear(
            in_features=hidden_size,
            out_features=1,
        )

    def forward(self, q, k, v, pad_mask=None):
        """
        Args:
            q: shape is [batch_size, hidden_size]
            k: shape is [batch_size, max_len, hidden_size]
            v: shape is [batch_size, max_len, hidden_size]
            pad_mask: shape is [batch_size, max_len], 0 for not padding and 1 for padding
        Returns:
            context: shape is [batch_size, hidden_size]
            weights: shape is [batch_size, max_len]
        """
        assert q.shape[1] == k.shape[2]
        assert k.shape[1] == v.shape[1]

        mid = torch.tanh(k + self.linear_q(q).unsqueeze(1))
        # mid = torch.tanh(self.linear_k(k) + self.linear_q(q).unsqueeze(1))
        # shape of mid is [batch_size, max_len, hidden_size]
        logits = self.linear_attn(mid).squeeze(2)
        # logits = torch.sum(mid, dim=2)
        # shape of logits is [batch_size, max_len]

        if pad_mask is not None:
            logits[pad_mask] = float('-inf')
        weights = F.softmax(logits, dim=1)

        context = torch.bmm(weights.unsqueeze(1), v).squeeze(1)
        # shape of context is [batch_size, v_size]
        return context, weights

class Noise(nn.Module):
    """
    Module for adding noise
    ref:
        https://github.com/facebookresearch/UnsupervisedMT/blob/master/NMT/src/trainer.py
    """
    def __init__(self, config):
        super(Noise, self).__init__()

        self.mask_prob = getattr(config, "mask_prob", 0.0)
        self.drop_prob = getattr(config, "drop_prob", 0.0)
        self.shuffle_weight = getattr(config, "shuffle_weight", 0.0)

        self.backup_mask_prob = self.mask_prob
        self.backup_drop_prob = self.drop_prob
        self.backup_shuffle_weight = self.shuffle_weight

        assert 0 <= self.mask_prob <= 1
        assert 0 <= self.drop_prob <= 1
        assert 0 <= self.shuffle_weight

        self.pad_id = config.pad_id
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.mask_id = getattr(config, "mask_id", None)

        self.device = config.device
    
    def set_mask_id(self, mask_id):
        self.mask_id = mask_id
    
    def zero_noise(self):
        self.mask_prob = 0
        self.drop_prob = 0
        self.shuffle_weight = 0
    
    def reset_noise(self):
        self.mask_prob = self.backup_mask_prob
        self.drop_prob = self.backup_drop_prob
        self.shuffle_weight = self.backup_shuffle_weight

    def mask(self, x, pad_sos_eos_mask):
        """
        WARNNING: this is a IN-PLACE op.
        Args:
            x: shape is [batch_size, max_x_len]
            pad_sos_eos_mask: shape is [batch_size, max_len], 0 for not padding and 1 for padding, sos and eos
        Returns:
            x: shape is [batch_size, max_x_len]
        """
        if self.mask_prob == 0:
            return x
        
        batch_size, max_x_len = x.shape
        assert (x[:, 0] == self.sos_id).sum() == batch_size
        # np.random.rand(d1, d2, ...) for [0,1)
        mask_mask = torch.tensor(np.random.rand(batch_size, max_x_len) < self.mask_prob).to(self.device)
        mask_mask *= torch.logical_not(pad_sos_eos_mask)  # and
        x[mask_mask] = self.mask_id
        return x
    
    def drop(self, x, x_len, pad_sos_eos_mask):
        """
        Args:
            x: shape is [batch_size, max_x_len]
            x_len: shape is [batch_size]
        Returns:
            x: shape is [batch_size, new_max_x_len]
            x_len: shape is [batch_size]
        """
        if self.drop_prob == 0:
            return x, x_len, pad_sos_eos_mask
        
        batch_size, max_x_len = x.shape
        assert (x[:, 0] == self.sos_id).sum() == batch_size
        # np.random.rand(d1, d2, ...) for [0,1)
        keep_mask = torch.tensor(np.random.rand(batch_size, max_x_len) >= self.drop_prob).to(self.device)
        keep_mask *= torch.logical_not(pad_sos_eos_mask)  # drop pad, sos and eos
        keep_mask[:, 0] = 1  # keep sos
        keep_mask += x == self.eos_id  # or op. to keep eos
        
        new_batch = []
        new_lens = []
        new_max_len = 0
        for tokens, mask in zip(x, keep_mask):
            new_tokens = tokens[mask]
            new_len = len(new_tokens)
            new_lens.append(new_len)
            new_max_len = max(new_max_len, new_len)
            new_batch.append(new_tokens)
        
        new_x = torch.LongTensor(batch_size, new_max_len).fill_(self.pad_id).to(self.device)
        new_x_len = torch.LongTensor(new_lens).to(self.device)
        for i, v in enumerate(zip(new_batch, new_x_len)):
            tokens, l = v
            new_x[i, :l].copy_(tokens)
        
        new_pad_sos_eos_mask = (new_x==self.sos_id)+(new_x==self.eos_id)+(new_x==self.pad_id)

        return new_x, new_x_len, new_pad_sos_eos_mask

    def shuffle(self, x, pad_sos_eos_mask):
        """
        only works when shuffle_weights is greater than 1
        Args:
            x: shape is [batch_size, max_x_len]
        Returns:
            x: shape is [batch_size, new_max_x_len]
        """
        if self.shuffle_weight <= 1:
            return x
        
        batch_size, max_x_len = x.shape
        assert (x[:, 0] == self.sos_id).sum() == batch_size
        
        weights = torch.FloatTensor(np.arange(start=0, stop=max_x_len)).unsqueeze(0).expand(batch_size, -1).clone().to(self.device)
        weights += torch.tensor(np.random.uniform(0, self.shuffle_weight, size=(batch_size, max_x_len))).to(self.device)

        weights[pad_sos_eos_mask] = max_x_len + self.shuffle_weight + 1  # fix eos and pad pos.
        weights[:, 0] = -1  # fix sos pos.

        permutation = torch.argsort(weights, dim=1)
        new_x = torch.gather(x, dim=1, index=permutation)
        return new_x

class Hypothesis:
    def __init__(self, out, hidden, last_ctx, score):
        self.out = out
        self.hidden = hidden
        self.last_ctx = last_ctx
        self.score = score

class SelfAttnRNNClassifier(nn.Module):
    """
    RNN Classifier
    """
    def __init__(self, config, embedding=None):
        super(SelfAttnRNNClassifier, self).__init__()

        self.cell = getattr(config, "cell", "LSTM").upper()
        self.bidirectional = getattr(config, "enc_bidirectional", True)
        pad_id = getattr(config, "pad_id", 0)
        # dropout = getattr(config, "dropout", 0.0)
        dropout = 0.0
        num_layers = getattr(config, "enc_num_layers", 1)

        self.hidden_size = config.hidden_size
        # attention_size = getattr(config, "attention_size", self.hidden_size)

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
            dropout=0.0,
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

        # self.dropout_layer = nn.Dropout(dropout)

        self.linear_emb = nn.Linear(
            in_features=config.emb_size,
            out_features=self.hidden_size,
        )

        self.linear_q = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
        )
        # self.linear_k = nn.Linear(
        #     in_features=self.hidden_size,
        #     out_features=self.hidden_size,
        # )
        self.linear_attn = nn.Linear(
            in_features=self.hidden_size,
            out_features=1,
        )
        # self.v_attn = nn.Parameter(torch.FloatTensor(self.hidden_size)).to(config.device)
        # print("self.v_attn.shape is", self.v_attn.shape)  # torch.Size([1, 1, 256])
        # self.q_attn = nn.Parameter(torch.FloatTensor(1,self.hidden_size)).to(config.device)

        self.output_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=config.num_labels,
        )

        self.device = config.device

    def freeze_emb(self):
        print("[SelfAttnRNNClassifier.freeze_emb] freeze all param in embedding")
        for param in self.embedding.parameters():
            param.requires_grad = False
    
    def unfreeze_emb(self):
        print("[SelfAttnRNNClassifier.freeze_emb] unfreeze all param in embedding")
        for param in self.embedding.parameters():
            param.requires_grad = True

    def attention(self, q, k, v, pad_sos_eos_mask):
        """
        Args:
            q: shape is [batch_size, q_size]
            k: shape is [batch_size, max_len, q_size]
            v: shape is [batch_size, max_len, v_size]
            pad_sos_eos_mask: shape is [batch_size, max_len], 1 for pad, sos and eos, 0 for others
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
        # print("[SelfAttnRNNClassifier.attention] logits is", logits)
        # print("[SelfAttnRNNClassifier.attention] pad_sos_eos_mask is", pad_sos_eos_mask)

        # logits = torch.sum(mid, dim=2)
        # shape of logits is [batch_size, max_len]
        
        for i in range(pad_sos_eos_mask.shape[0]):
            if pad_sos_eos_mask[i].all():
                pad_sos_eos_mask[i,1] = False
        logits[pad_sos_eos_mask] = float('-inf')
        # print("[SelfAttnRNNClassifier.attention] logits is", logits)

        weights = F.softmax(logits, dim=1)
        # print("[SelfAttnRNNClassifier.attention] torch.sum(weights, dim=1) is", torch.sum(weights, dim=1))
        # print("[SelfAttnRNNClassifier.attention] (torch.sum(weights, dim=1)-1.0 is", (torch.sum(weights, dim=1)-1.0))

        # with torch.no_grad():
        #     _sum = torch.sum(weights, dim=1)
        #     _sum[torch.isnan(_sum)] = 1.0
        #     assert (_sum-1.0 < 1e-4).all()

        context = torch.bmm(weights.unsqueeze(1), v).squeeze(1)
        # shape of context is [batch_size, v_size]

        return context, weights

    def forward(self, src, src_len, pad_sos_eos_mask, hidden=None, soft=False):
        """
        Args:
            src: shape is [batch_size, max_src_len]
            src_len: shape is [batch_size]
            pad_sos_eos_mask: shape is [batch_size, max_len], 1 for pad, sos and eos, 0 for others
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
        # embedded = self.dropout_layer(embedded)
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
                hidden[0], k, k, pad_sos_eos_mask,
            )
        else:
            hidden = hidden.view(-1, num_directions, batch_size, self.hidden_size)[-1]
            hidden = self.hidden_bridges[0](hidden.transpose(0,1).reshape(-1 ,num_directions*self.hidden_size))
            
            context, weights = self.attention(
                hidden, k, k, pad_sos_eos_mask,
            )
        
        logits = self.output_layer(context)
        # shape of logits is [batch_size, num_label]

        # Return output and final hidden state
        return logits, weights

class TransferModel(nn.Module):
    def __init__(self, config=None, embedding=None):
        super().__init__()
        if not config:
            return

        self.classifier = SelfAttnRNNClassifier(config)

        self.encoder = RNNEncoder(config, embedding)
        self.decoder = AttnRNNDecoder(config, self.encoder.embedding)

        self.bridge_dec_init = nn.Sequential(
            nn.Linear(
                in_features=config.hidden_size,  # only content hidden
                out_features=config.hidden_size,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
            )
        )

        self.noise = Noise(config)

        self.device = config.device
        self.cell = config.cell
        
        self.pad_id = config.pad_id
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.dec_num_layers = config.dec_num_layers
        
        self.cross_entropy_vocab = nn.CrossEntropyLoss(reduction='none', ignore_index=config.pad_id)
        self.cross_entropy_lab = nn.CrossEntropyLoss(reduction='none')
        self.distance = nn.MSELoss(reduction='none')

        self.freeze_cls_flag = False

    def freeze_cls(self):
        print("[TransferModel.freeze_cls] freeze all param in classifier")
        self.freeze_cls_flag = True
        for param in self.classifier.parameters():
            param.requires_grad = False

    def freeze_emb(self):
        print("[TransferModel.freeze_emb] freeze all param in embedding")
        self.encoder.freeze_emb()
        self.decoder.freeze_emb()
    
    def unfreeze_emb(self):
        print("[TransferModel.unfreeze_emb] unfreeze all param in embedding")
        self.encoder.unfreeze_emb()
        self.decoder.unfreeze_emb()
    
    def eval(self):
        self.noise.zero_noise()
        super().eval()
    
    def train(self, mode=True):
        if mode:
            self.noise.reset_noise()
        super().train(mode)
    
    def add_noise(self, src, src_len, pad_sos_eos_mask):
        """
        Args:
            src: shape is [batch_size, max_src_len]
            src_len: shape is [batch_size]
            pad_sos_eos_mask: shape is [batch_size, max_len],
                              1 for padding, sos and eos
        """
        if self.noise.mask_prob > 0:
            # note that self.noise.mask will change src in-place if mask prob > 0
            src = self.noise.mask(src.clone(), pad_sos_eos_mask)

        src, src_len, pad_sos_eos_mask = self.noise.drop(src, src_len, pad_sos_eos_mask)
        pad_mask = src==self.pad_id

        src = self.noise.shuffle(src, pad_sos_eos_mask)

        return src, src_len, (pad_mask, pad_sos_eos_mask)

        

    def forward(self, src, src_len, lab, pad_mask):
        """
        Args:
            src: shape is [batch_size, max_src_len]
            src_len: shape is [batch_size]
            lab: shape is [batch_size]
            pad_mask: shape is [batch_size, max_len], 1 for padding and 0 for not padding
        Returns:
            rec_loss: shape is [batch_size]
            align: list of tuple, each tuple has 4 alignments, each
                              has shape of [batch_size, max_src_len]
        """
        batch_size = src.shape[0]
        inv_lab = 1-lab

        self.decoder.set_style_emb(lab, inv_lab)

        # mask out [sos] [eos] tokens
        _pad_mask = pad_mask[:, 2:]
        tmp = torch.ones((batch_size,1), dtype=bool).to(self.device)
        pad_sos_eos_mask = torch.cat([tmp, _pad_mask, tmp], dim=1)

        ########################################################################
        # classification
        with torch.no_grad():
            cls_logits, style_alpha = self.classifier(src, src_len, pad_sos_eos_mask)
            # shape of cls_logits is [batch_size, num_label]
            # shape of style_alpha is [batch_size, max_src_len]
            src_cls_loss = self.cross_entropy_lab(cls_logits, lab)

            content_alpha = 1.0 - style_alpha
            content_alpha[pad_sos_eos_mask] = 0.0
            
            content_alpha /= (src_len-3+1e-4).unsqueeze(-1)
            
            # assert (torch.sum(content_alpha, dim=1) - 1.0 < 1e-4).all()
        ########################################################################

        src = src.transpose(0,1)
        # shape of src is [max_src_len, batch_size]

        ########################################################################
        # auto-encoder
        noise_src, noise_src_len, noise_mask_tuple = self.add_noise(src.transpose(0,1), src_len, pad_sos_eos_mask)
        noise_pad_mask, noise_pad_sos_eos_mask = noise_mask_tuple

        # ======================================================================
        # sort
        sorted_noise_src_len, noise_indices = torch.sort(noise_src_len, dim=0, descending=True)
        _, resorted_noise_indices = torch.sort(noise_indices, dim=0)
        sorted_noise_src = torch.index_select(noise_src, dim=0, index=noise_indices)
        sorted_noise_pad_sos_eos_mask = torch.index_select(noise_pad_sos_eos_mask, dim=0,
            index=noise_indices)
        # ======================================================================

        sorted_noise_src = sorted_noise_src.transpose(0,1)
        # shape of sorted_noise_src is [max_src_len, batch_size]
        sorted_noise_enc_out, sorted_noise_enc_hidden = self.encoder(sorted_noise_src, sorted_noise_src_len)
        # shape of enc_out is [max_src_len, batch_size, hidden_size]
        if self.cell.upper() == "LSTM":
            sorted_noise_enc_hidden = sorted_noise_enc_hidden[0]
        
        # ======================================================================
        # re-sort
        noise_enc_out = torch.index_select(sorted_noise_enc_out, dim=1,
            index=resorted_noise_indices)
        noise_enc_hidden = torch.index_select(sorted_noise_enc_hidden, dim=0,
            index=resorted_noise_indices)
        # ======================================================================


        noise_dec_init = self.bridge_dec_init(
            noise_enc_hidden
        ).unsqueeze(0).expand(self.dec_num_layers, -1, -1)
        # print("[HardDualAttnAutoEncoder.forward] dec_init.shape is", dec_init.shape)  #  torch.Size([1, 64, 512])

        if self.cell.upper() == "LSTM":
            noise_dec_init_h = noise_dec_init
            noise_dec_init_c = torch.tanh(noise_dec_init_h)
            noise_dec_init = (noise_dec_init_h, noise_dec_init_c)
        
        noise_decoder_out, noise_align = self.decoder(src, src_len,
            lab, noise_enc_out,
            noise_pad_mask, hidden=noise_dec_init,
            transfer=False)
        # shape of decoder_out is [max_trg_len-1, batch_size, vocab_size]
        # shape of align is a list of tuple of [batch_size, max_content_len] and
        # [batch_size, max_style_len], len of list is max_trg_len-1
        # decoder_out_ind = torch.argmax(decoder_out, dim=-1)

        # decoder_out = decoder_out.transpose(0,1)
        
        trg = src[1:]  # no [sos]
        trg_len = src_len - 1
        org_shape = trg.shape
        _trg = trg.contiguous().view(-1)
        _noise_decoder_out = noise_decoder_out.contiguous().view(-1, noise_decoder_out.shape[-1])
        
        rec_loss = self.cross_entropy_vocab(_noise_decoder_out, _trg).reshape(org_shape).sum(dim=0) / trg_len
        ########################################################################


        ########################################################################
        # encoder (no noise)
        enc_out, enc_hidden = self.encoder(src, src_len)
        # shape of enc_out is [max_src_len, batch_size, hidden_size]
        if self.cell.upper() == "LSTM":
            enc_hidden = enc_hidden[0]
        
        dec_init = self.bridge_dec_init(
            enc_hidden
        ).unsqueeze(0).expand(self.dec_num_layers, -1, -1)

        if self.cell.upper() == "LSTM":
            dec_init_h = dec_init
            dec_init_c = torch.tanh(dec_init_h)
            dec_init = (dec_init_h, dec_init_c)
        ########################################################################


        ########################################################################
        # soft decode for LM
        # shape of src is [max_src_len, batch_size]
        soft_out, soft_out_len, soft_out_align\
            = self.decoder.soft_decode(lab, enc_out,
                pad_mask, hidden=dec_init)
        # shape of soft_out is [max_len, batch_size, vocab_size]
        # shape of align is a list of tuple of [batch_size, max_content_len] and
        # [batch_size, max_style_len], len of list is max_trg_len-1
        # ======================================================================
        # sort
        sorted_soft_out_len, soft_out_indices = torch.sort(soft_out_len,
            dim=0, descending=True)
        _, resorted_soft_out_indices = torch.sort(soft_out_indices, dim=0)
        sorted_soft_out = torch.index_select(soft_out, dim=1, index=soft_out_indices)
        # all the lab are the same in a batch, so we do not need to sort lab
        sorted_soft_lab = torch.index_select(inv_lab, dim=0, index=soft_out_indices)
        # ======================================================================
        
        tmp = torch.arange(sorted_soft_out.shape[0]).to(self.device)
        sorted_soft_out_pad_sos_eos_mask = tmp.unsqueeze(0).expand((sorted_soft_out.shape[1], -1))\
            >= (sorted_soft_out_len.unsqueeze(1).expand((-1, sorted_soft_out.shape[0]))-1)
        sorted_soft_out_pad_sos_eos_mask[:, 0] = True  # set pos. of sos to be True

        sorted_soft_out_cls_logits, sorted_soft_out_style_alpha\
            = self.classifier(sorted_soft_out.transpose(0,1), sorted_soft_out_len,
                sorted_soft_out_pad_sos_eos_mask, soft=True)
        # shape of cls_logits is [batch_size, num_label]
        # shape of style_alpha is [batch_size, max_src_len]

        # ======================================================================
        # style_alpha and content_alpha
        with torch.no_grad():
            sorted_soft_out_style_alpha = sorted_soft_out_style_alpha.clone().detach()
            sorted_soft_out_style_alpha[torch.isnan(sorted_soft_out_style_alpha)] = 0.0
            sorted_soft_out_content_alpha = 1.0 - sorted_soft_out_style_alpha
            sorted_soft_out_content_alpha[sorted_soft_out_pad_sos_eos_mask] = 0.0
            
            sorted_soft_out_content_alpha /= (sorted_soft_out_len-3+1e-4).unsqueeze(-1)
            for i, l in enumerate(sorted_soft_out_len):
                if l==2:
                    sorted_soft_out_content_alpha[i, 1] = 1.0
            # assert (torch.sum(sorted_soft_out_content_alpha, dim=1) - 1.0 < 1e-4).all()
        # ======================================================================

        # ======================================================================
        # re-sort
        soft_out_cls_logits = torch.index_select(sorted_soft_out_cls_logits, dim=0, index=resorted_soft_out_indices)
        soft_out_content_alpha = torch.index_select(sorted_soft_out_content_alpha, dim=0, index=resorted_soft_out_indices)
        # ======================================================================

        soft_out_cls_loss = self.cross_entropy_lab(soft_out_cls_logits, inv_lab)

        ########################################################################

        ########################################################################
        # align content
        
        src_emb = self.encoder.embedding(src).transpose(0,1)
        # shapes of src_emb is [batch_size, max_src_len, emb_size]
        soft_out_emb = (soft_out @ self.encoder.embedding.weight).transpose(0,1)
        # shapes of src_emb is [batch_size, max_len, emb_size]
        src_content = (src_emb * content_alpha.unsqueeze(-1)).sum(1)
        soft_out_content = (soft_out_emb * soft_out_content_alpha.unsqueeze(-1)).sum(1)

        content_align_loss = self.distance(soft_out_content, src_content).sum(1)
        ########################################################################


        ########################################################################
        # greedy decode for BT
        # shape of src is [max_src_len, batch_size]
        with torch.no_grad():
            out, out_len, out_logits, out_align = self.decoder.sample_decode(
                lab, enc_out, pad_mask, hidden=dec_init, greedy=True)
            # shape of out is [max_len, batch_size]
            # shape of out_len is [batch_size]
            # shape of out_logits is [max_len, batch_size, vocab_size]
            # shape of out_align is a list of tuple of [batch_size, max_content_len] and
            # [batch_size, max_style_len], len of list is max_len

            # ==================================================================
            # sort
            sorted_out_len, out_indices = torch.sort(out_len,
                dim=0, descending=True)
            _, resorted_out_indices = torch.sort(out_indices, dim=0)
            sorted_out = torch.index_select(out, dim=1, index=out_indices)
            sorted_out_inv_lab = torch.index_select(inv_lab, dim=0, index=out_indices)
            # ==================================================================

            tmp = torch.arange(sorted_out.shape[0]).to(self.device)
            sorted_out_pad_sos_eos_mask = tmp.unsqueeze(0).expand((sorted_out.shape[1], -1))\
                >= (sorted_out_len.unsqueeze(1).expand((-1, sorted_out.shape[0]))-1)
            sorted_out_pad_sos_eos_mask[:, 0] = True  # set pos. of sos to be True

            sorted_out_cls_logits, sorted_out_style_alpha =\
                self.classifier(sorted_out.transpose(0,1), sorted_out_len,
                    sorted_out_pad_sos_eos_mask)
            # shape of cls_logits is [batch_size, num_label]
            # shape of style_alpha is [batch_size, max_src_len]
            
            sorted_out_cls_loss = self.cross_entropy_lab(sorted_out_cls_logits, sorted_out_inv_lab)
            
            
        sorted_out_enc_out, sorted_out_hidden = self.encoder(
            sorted_out, sorted_out_len)
        # shape of sorted_out_enc_out is [max_src_len, batch_size, hidden_size]
        if self.cell.upper() == "LSTM":
            sorted_out_hidden = sorted_out_hidden[0]
        # shape of sorted_out_hidden is [batch_size, hidden_size]

        # ======================================================================
        # re-sort
        out_enc_out = torch.index_select(sorted_out_enc_out, dim=1,
            index=resorted_out_indices)
        out_hidden = torch.index_select(sorted_out_hidden, dim=0,
            index=resorted_out_indices)
        out_style_alpha = torch.index_select(sorted_out_style_alpha, dim=0,
            index=resorted_out_indices)
        out_cls_loss = torch.index_select(sorted_out_cls_loss, dim=0,
            index=resorted_out_indices)
        # ======================================================================
        
        dec_init_out = self.bridge_dec_init(
            out_hidden
        ).unsqueeze(0).expand(self.dec_num_layers, -1, -1)
        if self.cell.upper() == "LSTM":
            dec_init_out_h = dec_init_out
            dec_init_out_c = torch.tanh(dec_init_out_h)
            dec_init_out = (dec_init_out_h, dec_init_out_c)
        out_pad_mask = out.transpose(0,1)==self.pad_id

        # shape of src is [max_src_len, batch_size]
        bt_logits, bt_align = self.decoder(src, src_len, lab, out_enc_out,
            out_pad_mask, hidden=dec_init_out, transfer=False)
        # shape of bt_logits is [max_src_len-1, batch_size, vocab_size]
        _bt_logits = bt_logits.contiguous().view(-1, bt_logits.shape[-1])
        bt_loss = self.cross_entropy_vocab(_bt_logits, _trg).reshape(org_shape).sum(dim=0) / trg_len
        ########################################################################

        self.decoder.drop_style_emb()

        return (rec_loss, bt_loss,
            src_cls_loss, soft_out_cls_loss, out_cls_loss, content_align_loss),\
            (out.transpose(0,1), out_len), out_align
