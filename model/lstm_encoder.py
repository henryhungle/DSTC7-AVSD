# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import pdb
from torch.nn.utils.weight_norm import weight_norm
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import BertModel, OpenAIGPTModel, GPT2Model, TransfoXLModel 

class LSTMEncoder(nn.Module):

    def __init__(self, n_layers, in_size, out_size, embed_size, dropout=0.5, initialEmbW=None, rnn_type='lstm', attention=None, q_size=-1, embedding_init=None, weights_init=None, elmo_init=False, elmo_num_outputs=1, finetune_elmo=False,
        bert_init=False, bert_model=None, finetune_bert=False,
        add_word_emb=True):
        """Initialize encoder with structure parameters
        Args:
            n_layers (int): Number of layers.
            in_size (int): Dimensionality of input vectors.
            out_size (int) : Dimensionality of hidden vectors to be output.
            embed_size (int): Dimensionality of word embedding.
            dropout (float): Dropout ratio.
        """
        # TODO 
        conv_out_size = 512

        super(LSTMEncoder, self).__init__()
        self.embed =nn.Embedding(in_size, embed_size)
        if embedding_init is not None: 
            self.embed.weight.data.copy_(torch.from_numpy(embedding_init))
        elif weights_init is not None: 
            self.embed.weight.data.copy_(torch.from_numpy(weights_init['embed']))
        self.elmo_init = elmo_init
        self.bert_init = bert_init
        self.bert_model = bert_model 
        self.add_word_emb=add_word_emb
        if elmo_init:
            options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, elmo_num_outputs, requires_grad=finetune_elmo)
            elmo_layer = [nn.Linear(elmo_num_outputs*1024, out_size), nn.ReLU()]
            self.elmo_layer = nn.Sequential(*elmo_layer)
        elif bert_init:
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
        if rnn_type == 'lstm': 
            self.lstm = nn.LSTM(embed_size, out_size, n_layers, batch_first=True, dropout=dropout)
        elif rnn_type == 'gru': 
            self.lstm = nn.GRU(embed_size, out_size, n_layers, batch_first=True, dropout=dropout)
        self.attention = attention  
        if attention == 'conv' or attention == 'conv_sum':
            conv_in_size = out_size
            self.conv1 = nn.Conv1d(in_channels=conv_in_size, out_channels=conv_out_size,kernel_size=1,padding=0)
            self.conv2 = nn.Conv1d(in_channels=conv_out_size, out_channels=2,kernel_size=1,padding=0)   
            if weights_init is not None:
                self.conv1.weight.data.copy_(torch.from_numpy(weights_init['conv1']))
                self.conv2.weight.data.copy_(torch.from_numpy(weights_init['conv2']))
        elif attention == 'c_conv_sum':
            hidden_size = 512
            conv_hidden_size = 256
            layers = [
                weight_norm(nn.Linear(out_size, hidden_size), dim=None),
                nn.ReLU()
            ]
            self.c_fa = nn.Sequential(*layers) 
            layers = [ 
                weight_norm(nn.Linear(q_size, hidden_size), dim=None),
                nn.ReLU()
            ]
            self.q_fa = nn.Sequential(*layers)
            layers = [
                nn.Conv2d(in_channels=hidden_size, out_channels=conv_hidden_size, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=conv_hidden_size, out_channels=1, kernel_size=1)
            ]
            self.cq_att = nn.Sequential(*layers)
            if weights_init is not None:
                self.c_fa[0].weight.data.copy_(torch.from_numpy(weights_init['c_fa']))
                self.q_fa[0].weight.data.copy_(torch.from_numpy(weights_init['q_fa']))
                self.cq_att[0].weight.data.copy_(torch.from_numpy(weights_init['cq_att_conv1']))
                self.cq_att[2].weight.data.copy_(torch.from_numpy(weights_init['cq_att_conv2']))
        

    def q_attention(self, embedding): 
        # embedding: (bsize, seq_len, hidden_dim)
        embedding_reshape = embedding.permute(0,2,1)
        # embedding: (bsize, hidden_dim, seq_len)
        conv1 = self.conv1(embedding_reshape)
        relu = F.relu(conv1)
        conv2 = self.conv2(relu)
        att_softmax = F.softmax(conv2, dim=2)
        if self.attention == 'conv_sum':
            feature = torch.bmm(att_softmax, embedding)
            feature_concat = feature.view(embedding.shape[0], -1)
        elif self.attention == 'conv':
            att_softmax_1 = att_softmax[:,0,:].unsqueeze(2).expand_as(embedding)
            att_softmax_2 = att_softmax[:,1,:].unsqueeze(2).expand_as(embedding)
            feature_concat = torch.cat([embedding*att_softmax_1, embedding*att_softmax_2], dim=2)
        return feature_concat

    def c_attention(self, c, q):
        seq_len = c.shape[1]
        c_fa = self.c_fa(c)
        q_fa = self.q_fa(q)
        q_fa_expand = torch.unsqueeze(q_fa,1).expand(-1,seq_len,-1)
        joint_feature = c_fa * q_fa_expand 
        joint_feature_reshape = torch.unsqueeze(joint_feature.permute(0,2,1),3)
        raw_attention = self.cq_att(joint_feature_reshape)
        raw_attention = torch.squeeze(raw_attention,3).permute(0,2,1)
        attention = F.softmax(raw_attention, dim=1).expand_as(c)
        c_feature = torch.sum(attention * c, dim=1)
        return c_feature

    def __call__(self, s, xs, states_att=False, q=None, context_x=None, **kwargs):
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
        if len(xs) != 0:
            sections = np.array([len(x) for x in xs], dtype=np.int32)
            aa = torch.cat(xs, 0)
            bb = self.embed(torch.tensor(aa, dtype=torch.long).cuda())
            cc = sections.tolist()
            wj = torch.split(bb, cc,dim=0)
            wj = list(wj)
            #sorting
            sort_wj=[]
            cc = torch.cuda.LongTensor(sections)
            cc, perm_index = torch.sort(cc, 0, descending=True)
            sort_wj.append([wj[i] for i in perm_index])
            padded_wj = nn.utils.rnn.pad_sequence(sort_wj[0], batch_first=True)
            packed_wj = nn.utils.rnn.pack_padded_sequence(padded_wj, tuple(cc.data), batch_first=True)          
        else:
            hx = [ self.embed(xs[0]) ]
        
        if s is not None:
            if self.rnn_type == 'lstm':
                ys, _ = self.lstm(packed_wj, (s[0], s[1]))
            elif self.rnn_type == 'gru':
                ys, _ = self.lstm(packed_wj, s)
        else:
            ys, _ = self.lstm(packed_wj) 
        #resorting
        ys = nn.utils.rnn.pad_packed_sequence(ys, batch_first=True)[0]

        if self.elmo_init:
            sorted_context_x = [context_x[i] for i in perm_index]
            char_ids = batch_to_ids(sorted_context_x).cuda()
            elmo_emb = self.elmo(char_ids)['elmo_representations'] 
            elmo_emb = torch.cat(elmo_emb, -1)
            elmo_emb = self.elmo_layer(elmo_emb)
            if hasattr(self, 'add_word_emb') and not self.add_word_emb:
                ys = elmo_emb
            else:
                ys = ys + elmo_emb
        elif self.bert_init:
            sorted_context_x = torch.cat([context_x[i].unsqueeze(0) for i in perm_index], 0)
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
            
        if self.attention == 'conv' or self.attention == 'conv_sum':
            ys = self.q_attention(ys)
        elif self.attention == 'c_conv_sum': 
            ys = self.c_attention(ys, q)
 
        if len(xs)>1:
            if self.attention is not None and (self.attention == 'conv_sum' or self.attention == 'c_conv_sum'):
                decoded = ys
            else:
                if states_att: 
                    decoded = ys 
                else:
                    idx = (cc - 1).view(-1, 1).expand(ys.size(0), ys.size(2)).unsqueeze(1)
                    idx = torch.tensor(idx, dtype=torch.long)
                    decoded = ys.gather(1, idx.cuda()).squeeze()
            # restore the sorting
            cc2, perm_index2 = torch.sort(perm_index, 0)
            if states_att:
                odx = perm_index2.view(-1, 1, 1).expand_as(ys)
            else:
                odx = perm_index2.view(-1, 1).expand(ys.size(0), ys.size(-1))
            decoded = decoded.gather(0, odx.cuda())
        else:
            if self.attention is not None and (self.attention == 'conv_sum' or self.attention == 'c_conv_sum'):
                decoded = ys 
            else:
                if states_att: 
                    decoded = ys 
                else:
                    decoded = ys[:,-1,:]

        return decoded

