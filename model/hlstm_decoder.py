# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import six
import pdb 
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import BertModel, OpenAIGPTModel, GPT2Model, TransfoXLModel

class HLSTMDecoder(nn.Module):

    frame_based = False
    take_all_states = False

    def __init__(self, n_layers, in_size, out_size, embed_size, in_size_hier, hidden_size, proj_size, dropout=0.5, initialEmbW=None, independent=False, rnn_type='lstm', classifier='baseline', states_att=False, state_size=-1, embedding_init=None, weights_init=None,
        elmo_init=False, elmo_num_outputs=1, finetune_elmo=False,
        bert_init=False, bert_model=None, finetune_bert=False,
        add_word_emb=True, pretrained_all=True):
        """Initialize encoder with structure parameters

        Args:
            n_layers (int): Number of layers.
            in_size (int): Dimensionality of input vectors.
            out_size (int): Dimensionality of output vectors.
            embed_size (int): Dimensionality of word embedding.
            hidden_size (int) : Dimensionality of hidden vectors.
            proj_size (int) : Dimensionality of projection before softmax.
            dropout (float): Dropout ratio.
        """
        #TODO 
        att_size = 128
        self.rnn_type = rnn_type 
        self.classifier = classifier 
        super(HLSTMDecoder, self).__init__()
        self.embed = nn.Embedding(in_size, embed_size)
        if embedding_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embedding_init))
        elif weights_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(weights_init['embed']))
        if rnn_type == 'lstm':
            self.lstm = nn.LSTM(embed_size+in_size_hier, hidden_size, n_layers, batch_first=True, dropout=dropout)
        elif rnn_type == 'gru': 
            self.lstm = nn.GRU(embed_size+in_size_hier, hidden_size, n_layers, batch_first=True, dropout=dropout)
        if weights_init is not None:
            lstm_wt = weights_init['lstm']
            for k,v in lstm_wt.items():
                self.lstm.__getattr__(k).data.copy_(torch.from_numpy(v))

        self.elmo_init = elmo_init
        self.bert_init = bert_init
        self.pretrained_all = pretrained_all
        self.bert_model = bert_model 
        self.add_word_emb=add_word_emb
        if False:
        #if pretrained_all and elmo_init:
            options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, elmo_num_outputs, requires_grad=finetune_elmo)
            elmo_layer = [nn.Linear(elmo_num_outputs*1024, out_size), nn.ReLU()]
            self.elmo_layer = nn.Sequential(*elmo_layer)
        elif False:
        #elif pretrained_all and bert_init:
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

        self.n_layers = n_layers
        self.dropout = dropout
        self.independent = independent
        self.states_att = states_att
        if states_att:
            self.ecW = nn.Linear(state_size, att_size)
            self.ysW = nn.Linear(hidden_size, att_size)
            hidden_size += state_size 

        if classifier == 'baseline': 
            layers = [
                nn.Linear(hidden_size, proj_size),
                nn.Linear(proj_size, out_size)
            ]
            self.y_classifier = nn.Sequential(*layers)
        elif classifier == 'weighted_norm':
            layers = [
                weight_norm(nn.Linear(hidden_size, proj_size), dim=None),
                nn.ReLU(),
                weight_norm(nn.Linear(proj_size, out_size), dim=None)
            ]
            self.y_classifier = nn.Sequential(*layers)
        elif classifier == 'logit':
            layers = [
                weight_norm(nn.Linear(hidden_size, proj_size), dim=None),
                nn.ReLU(),
                nn.Linear(proj_size, out_size)
            ]
            self.classifier_txt = nn.Sequential(*layers)
            layers = [
                weight_norm(nn.Linear(hidden_size, 2048), dim=None),
                nn.ReLU(),
                nn.Linear(2048, out_size)
            ]
            self.classifier_ft = nn.Sequential(*layers)
            if weights_init is not None:
                self.classifier_txt[0].weight.data.copy_(torch.from_numpy(weights_init['classifier_txt']))
                self.classifier_ft[0].weight.data.copy_(torch.from_numpy(weights_init['classifier_ft']))

    def states_attention(self, ec, ys):
        linear_ec = self.ecW(ec)
        linear_ys = self.ysW(ys)
        linear_ec_permute = linear_ec.permute(0,2,1)
        att_scores = torch.bmm(linear_ys, linear_ec_permute)
        att_scores = F.softmax(att_scores, dim=2).unsqueeze(3)
        ec_unsq = ec.unsqueeze(1)
        ec_expand = ec_unsq.expand(ec_unsq.shape[0], ys.shape[1], ec_unsq.shape[2], ec_unsq.shape[3])
        att_ec = ec_expand * att_scores 
        combined_ec = att_ec.sum(2)
        joint_ft = torch.cat((ys, combined_ec), dim=2)
        return joint_ft

    def __call__(self, s, hs, xs, ec=None, context_y=None):
        """Calculate all hidden states, cell states, and output prediction.

        Args:
            s (~chainer.Variable or None): Initial (hidden, cell) states.  If ``None``
                is specified zero-vector is used.
            hs (list of ~chianer.Variable): List of input state sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.
            xs (list of ~chianer.Variable): List of input label sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.
        Return:
            (hy,cy): a pair of hidden and cell states at the end of the sequence,
            y: a sequence of pre-activatin vectors at the output layer
 
        """
        if len(xs) > 1:
            sections = np.array([len(x) for x in xs], dtype=np.int32)
            aa = torch.cat(xs, 0)
            bb = self.embed(torch.tensor(aa, dtype=torch.long).cuda())
            cc = sections.tolist()
            hx = torch.split(bb, cc, dim=0)
        else:
            hx = [ self.embed(xs[0]) ]

        hxc = [ torch.cat((hx[i], hs[i].repeat(hx[i].shape[0], 1)), dim=1) for i in six.moves.range(len(hx))]

        sort_hxc = []
        cc = torch.from_numpy(sections)
        cc, perm_index = torch.sort(cc, 0, descending=True)
        sort_hxc.append([hxc[i] for i in perm_index])
        padded_hxc = nn.utils.rnn.pad_sequence(sort_hxc[0], batch_first=True)
        packed_hxc = nn.utils.rnn.pack_padded_sequence(padded_hxc, list(cc.data), batch_first=True)
        if s is None or (hasattr(self, 'independent') and self.independent):
            if self.rnn_type == 'lstm': 
                ys, (hy, cy) = self.lstm(packed_hxc)
            elif self.rnn_type == 'gru': 
                ys, hy = self.lstm(packed_hxc)
            ys = nn.utils.rnn.pad_packed_sequence(ys, batch_first=True)[0]

            # restore the sorting
            cc2, perm_index2 = torch.sort(perm_index, 0)
            odx = perm_index2.view(-1, 1).unsqueeze(1).expand(ys.size(0), ys.size(1), ys.size(2))
            ys2 = ys.gather(0, odx.cuda())
            if self.states_att:
                ys2 = self.states_attention(ec, ys)
            ys2_list=[]
            ys2_list.append([ys2[i,0:sections[i],:] for i in six.moves.range(ys2.shape[0])])
            
        if self.classifier == 'weighted_norm' or self.classifier == 'baseline':
            y = self.y_classifier(F.dropout(torch.cat(ys2_list[0],dim=0), p=self.dropout))
        elif self.classifier == 'logit':
            y_txt = self.classifier_txt(F.dropout(torch.cat(ys2_list[0],dim=0), p=self.dropout))
            y_ft = self.classifier_ft(F.dropout(torch.cat(ys2_list[0],dim=0), p=self.dropout))
            y = y_txt + y_ft 
            
        if self.rnn_type == 'lstm': 
            return (hy,cy),y
        elif self.rnn_type == 'gru':
            return hy, y 

    # interface for beam search
    def initialize(self, s, x, i):
        """Initialize decoder

        Args:
            s (any): Initial (hidden, cell) states.  If ``None`` is specified
                     zero-vector is used.
            x (~chainer.Variable or None): Input sequence
            i (int): input label.
        Return:
            initial decoder state
        """
        # LSTM decoder can be initialized in the same way as update()
        if len(x) > 1:
            self.hx = F.vstack([x[j][-1] for j in six.moves.range(len(x[1]))])
        else:
            self.hx = x
        if hasattr(self, 'independent') and self.independent:
            return self.update(None,i)
        else:
            return self.update(s,i)


    def update(self, s, i):
        """Update decoder state

        Args:
            s (any): Current (hidden, cell) states.  If ``None`` is specified 
                     zero-vector is used.
            i (int): input label.
        Return:
            (~chainer.Variable) updated decoder state
        """
        x = torch.cat((self.embed(i), self.hx), dim=1)
        if s is not None and len(s[0]) == self.n_layers*2:
            s = list(s)
            for m in (0,1):
                ss = []
                for n in six.moves.range(0,len(s[m]),2):
                    ss.append(F.concat((s[m][n],s[m][n+1]), axis=1))
                s[m] = F.stack(ss, axis=0)

        if len(i) != 0:
            xs = torch.unsqueeze(x,0)
        else:
            xs = [x]

        if s is not None:
            if self.rnn_type == 'lstm': 
                dy, (hy, cy) = self.lstm(xs, (s[0], s[1]))
            elif self.rnn_type == 'gru':
                dy, hy = self.lstm(xs, s[0])
        else:
            if self.rnn_type == 'lstm': 
                dy, (hy, cy) = self.lstm(xs)
            elif self.rnn_type == 'gru':
                dy, hy = self.lstm(xs)
        if self.rnn_type == 'lstm':
            return hy, cy, dy
        elif self.rnn_type == 'gru': 
            return hy, dy 

    def predict(self, s, ec=None):
        """Predict single-label log probabilities

        Args:
            s (any): Current (hidden, cell) states.
        Return:
            (~chainer.Variable) log softmax vector
        """
        if self.rnn_type == 'lstm':
            feature = s[2][0]
        elif self.rnn_type == 'gru':
            feature = s[1][0]
        if self.states_att:
            feature_unsq = feature.unsqueeze(0)
            feature = self.states_attention(ec, feature_unsq)
            feature = feature.squeeze(0)
        if self.classifier == 'weighted_norm' or self.classifier == 'baseline':
            y = self.y_classifier(feature)
        elif self.classifier == 'logit':
            y_txt =self.classifier_txt(feature)
            y_ft = self.classifier_ft(feature)
            y = y_txt + y_ft 
        return F.log_softmax(y, dim=1)

