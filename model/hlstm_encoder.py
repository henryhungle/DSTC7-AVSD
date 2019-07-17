# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import six
import scipy.io as sio
import pdb 
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import BertModel, OpenAIGPTModel, GPT2Model, TransfoXLModel

class HLSTMEncoder(nn.Module):

    def __init__(self, n_wlayers, n_slayers, in_size, out_size, embed_size, hidden_size, dropout=0.5, ignore_label=None, initialEmbW=None, independent=False, rnn_type='lstm', embedding_init=None, weights_init=None,
        elmo_init=False, elmo_num_outputs=1, finetune_elmo=False,
        bert_init=False, bert_model=None, finetune_bert=False,
        add_word_emb=True, pretrained_all=True, concat_his=False):
        """Initialize encoder with structure parameters
        Args:
            n_layers (int): Number of layers.
            in_size (int): Dimensionality of input vectors.
            out_size (int) : Dimensionality of hidden vectors to be output.
            embed_size (int): Dimensionality of word embedding.
            dropout (float): Dropout ratio.
        """

        super(HLSTMEncoder, self).__init__()
        self.embed = nn.Embedding(in_size, embed_size)
        if embedding_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embedding_init))
        elif weights_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(weights_init['embed']))
        if rnn_type == 'lstm':
            self.wlstm = nn.LSTM(embed_size, hidden_size, n_wlayers, batch_first=True, dropout=dropout)
            self.slstm = nn.LSTM(hidden_size, out_size, n_slayers, batch_first=True, dropout=dropout)
        elif rnn_type == 'gru':
            self.wlstm = nn.GRU(embed_size, hidden_size, n_wlayers,batch_first=True, dropout=dropout)
            self.slstm = nn.GRU(hidden_size, out_size, n_slayers, batch_first=True, dropout=dropout) 
        self.elmo_init = elmo_init
        self.bert_init = bert_init
        self.pretrained_all = pretrained_all
        self.concat_his = concat_his
        self.bert_model = bert_model 
        self.add_word_emb=add_word_emb
        if pretrained_all and elmo_init:
            options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, elmo_num_outputs, requires_grad=finetune_elmo)
            elmo_layer = [nn.Linear(elmo_num_outputs*1024, out_size), nn.ReLU()]
            self.elmo_layer = nn.Sequential(*elmo_layer)
        elif pretrained_all and bert_init:
            if 'bert' in bert_model:
                self.bert = BertModel.from_pretrained(bert_model)
            elif 'openai-gpt' in bert_model:
                self.bert = OpenAIGPTModel.from_pretrained(bert_model)
            elif 'gpt2' in bert_model:
                self.bert = GPT2Model.from_pretrained(bert_model)
            elif 'transfo-xl' in bert_model:
                self.bert = TransfoXLModel.from_pretrained(bert_model) 
            self.finetune_bert = finetune_bert
            if not finetune_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False 
            if bert_model in ['bert-base-uncased', 'openai-gpt', 'gpt2']: 
                bert_in = 768 
            elif bert_model in ['bert-large-uncased', 'gpt2-medium', 'transfo-xl-wt103']:
                bert_in = 1024
            bert_layer = [nn.Linear(bert_in, out_size), nn.ReLU()]
            self.bert_layer = nn.Sequential(*bert_layer)

        self.independent = independent
        self.rnn_type = rnn_type 

    def __call__(self, s, xs, context_x=None, **kwargs):
        """Calculate all hidden states and cell states.
        Args:
            s  (~chainer.Variable or None): Initial (hidden & cell) states. If ``None``
                is specified zero-vector is used.
            xs (list of ~chianer.Variable): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.
        Return:
            (hy,cy): a pair of hidden and cell states at the end of the sequence,
            ys: a hidden state sequence at the last layer
        """
        if hasattr(self, 'concat_his') and self.concat_his:
            xs = [xs]
            context_x = [context_x]
        # word level within sentence
        sx = []
        for l in six.moves.range(len(xs)):
            if len(xs[l]) != 0:
                sections = np.array([len(x) for x in xs[l]], dtype=np.int32)
                aa = torch.cat(xs[l], 0)
                bb = self.embed(torch.tensor(aa, dtype=torch.long).cuda())
                cc = sections.tolist()
                wj = torch.split(bb, cc, dim=0)
                wj = list(wj)
                # sorting
                sort_wj = []
                cc = torch.from_numpy(sections)
                cc, perm_index = torch.sort(cc, 0, descending=True)
                sort_wj.append([wj[i] for i in perm_index])
                padded_wj = nn.utils.rnn.pad_sequence(sort_wj[0], batch_first=True)
                packed_wj = nn.utils.rnn.pack_padded_sequence(padded_wj, list(cc.data), batch_first=True)
            else:
                xl = [ self.embed(xs[l][0]) ]
            if hasattr(self, 'independent') and self.independent:
                ys, why = self.wlstm(packed_wj)
            else:
                if l==0:
                    ys, why = self.wlstm(packed_wj)
                else:
                    if self.rnn_type == 'lstm': 
                        ys, why = self.wlstm(packed_wj, (why[0], why[1]))
                    elif self.rnn_type == 'gru': 
                        ys, why = self.wlstm(packed_wj, why)
            ys = nn.utils.rnn.pad_packed_sequence(ys, batch_first=True)[0]
            
            if self.pretrained_all and self.elmo_init:
                sorted_context_x = [context_x[l][i] for i in perm_index]
                char_ids = batch_to_ids(sorted_context_x).cuda()
                elmo_emb = self.elmo(char_ids)['elmo_representations'] 
                elmo_emb = torch.cat(elmo_emb, -1) 
                elmo_emb = self.elmo_layer(elmo_emb)
                if hasattr(self, 'add_word_emb') and not self.add_word_emb:
                    ys = elmo_emb
                else:
                    ys = ys + elmo_emb
            elif self.pretrained_all and self.bert_init:
                sorted_context_x = torch.cat([context_x[l][i].unsqueeze(0) for i in perm_index], 0)
                segments = torch.zeros(sorted_context_x.shape).long().cuda()
                if not self.finetune_bert:
                    with torch.no_grad():
                        if 'bert' in self.bert_model:
                            bert_emb, _ = self.bert(sorted_context_x.cuda(), segments)
                            bert_emb = bert_emb[-1]
                        elif 'openai-gpt' in self.bert_model:
                            bert_emb = self.bert(sorted_context_x.cuda())
                        elif 'gpt2' in self.bert_model or 'transfo-xl' in self.bert_model:
                            bert_emb, _ = self.bert(sorted_context_x.cuda())
                else:
                    if 'bert' in self.bert_model:
                        bert_emb, _ = self.bert(sorted_context_x.cuda(), segments)
                        bert_emb = bert_emb[-1]
                    elif 'openai-gpt' in self.bert_model:
                        bert_emb = self.bert(sorted_context_x.cuda())
                    elif 'gpt2' in self.bert_model or 'transfo-xl' in self.bert_model:
                        bert_emb, _ = self.bert(sorted_context_x.cuda())
    
                bert_emb = self.bert_layer(bert_emb)
                if hasattr(self, 'add_word_emb') and not self.add_word_emb:
                    ys = bert_emb
                else:
                    ys = ys + bert_emb
            
            if len(xs[l]) > 1:
                idx = (cc - 1).view(-1, 1).expand(ys.size(0), ys.size(2)).unsqueeze(1)
                idx = torch.tensor(idx, dtype=torch.long)
                decoded = ys.gather(1, idx.cuda()).squeeze()

                # restore the sorting
                cc2, perm_index2 = torch.sort(perm_index, 0)
                odx = perm_index2.view(-1, 1).expand(ys.size(0), ys.size(-1))
                decoded = decoded.gather(0, odx.cuda())
            else:
                decoded = ys[:, -1, :]

            sx.append(decoded)

        # sentence level
        sxs = torch.stack(sx, dim=0)
        sxs = sxs.permute(1,0,2)
        if s is not None:
            if self.rnn_type == 'lstm':
                sys, shy = self.slstm(sxs, (s[0], s[1]))
            elif self.rnn_type == 'gru':
                sys, shy = self.slstm(sxs, s)
        else:
            sys, shy = self.slstm(sxs)
        
        #pdb.set_trace()
        if self.rnn_type == 'gru':
            return shy
        elif self.rnn_type == 'lstm':
            return shy[0]

