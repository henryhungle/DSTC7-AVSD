#!/usr/bin/env python

import math
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import pdb

class MMEncoder(nn.Module):

    def __init__(self, in_size, out_size, enc_psize=[], enc_hsize=[], att_size=100,
                 state_size=100, attention='baseline', fusioning='baseline', att_hops=1):
        if len(enc_psize)==0:
            enc_psize = in_size
        if len(enc_hsize)==0:
            enc_hsize = [0] * len(in_size)

        # make links
        super(MMEncoder, self).__init__()
        # memorize sizes
        self.n_inputs = len(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.enc_psize = enc_psize
        self.enc_hsize = enc_hsize
        self.att_size = att_size
        self.state_size = state_size
        self.attention = attention 
        self.fusioning = fusioning 
        self.att_hops = att_hops 

        # encoder
        self.l1f_x = nn.ModuleList()
        self.l1f_h = nn.ModuleList()
        self.l1b_x = nn.ModuleList()
        self.l1b_h = nn.ModuleList()
        self.emb_x = nn.ModuleList()
        for m in six.moves.range(len(in_size)):
            self.emb_x.append(nn.Linear(self.in_size[m], self.enc_psize[m]))
            if enc_hsize[m] > 0:
                self.l1f_x.append(nn.Linear(enc_psize[m], 4 * enc_hsize[m]))
                self.l1f_h.append(nn.Linear(enc_hsize[m], 4 * enc_hsize[m], bias=False))
                self.l1b_x.append(nn.Linear(enc_psize[m], 4 * enc_hsize[m]))
                self.l1b_h.append(nn.Linear(enc_hsize[m], 4 * enc_hsize[m], bias=False))
        # temporal attention
        if attention == 'baseline': 
            self.atV = nn.ModuleList()
            self.atW = nn.ModuleList()
            self.atw = nn.ModuleList()
            self.lgd = nn.ModuleList()
            for m in six.moves.range(len(in_size)):
                enc_hsize_ = 2 * enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m]
                self.atV.append(nn.Linear(enc_hsize_, att_size))
                self.atW.append(nn.Linear(state_size, att_size))
                self.atw.append(nn.Linear(att_size, 1))
                self.lgd.append(nn.Linear(enc_hsize_, out_size))
        elif attention == 'conv' or attention == 'double_conv': 
            self.x_att = nn.ModuleList() 
            self.q_att = nn.ModuleList() 
            self.xq_att = nn.ModuleList()
            self.lgd = nn.ModuleList() 
            self.q_transform = nn.ModuleList()
            hidden_size = 5000 
            conv_hidden_size = 1000
            for m in range(len(in_size)):
                if attention == 'conv':
                    layers = [
                        weight_norm(nn.Linear(in_size[m], hidden_size), dim=None),
                        nn.ReLU()
                    ]
                elif attention == 'double_conv':
                    layers = [
                        weight_norm(nn.Linear(in_size[m], hidden_size), dim=None),
                        nn.ReLU(),  
                        weight_norm(nn.Linear(hidden_size, hidden_size), dim=None),
                        nn.ReLU()
                    ]
                self.x_att.append(nn.Sequential(*layers))
                if attention == 'conv':
                    layers = [
                        weight_norm(nn.Linear(state_size, hidden_size), dim=None), 
                        nn.ReLU()
                    ]
                elif attention == 'double_conv':
                    layers = [
                        weight_norm(nn.Linear(state_size, hidden_size), dim=None),
                        nn.ReLU(),
                        weight_norm(nn.Linear(hidden_size, hidden_size), dim=None),
                        nn.ReLU()
                    ]
                self.q_att.append(nn.Sequential(*layers))
                layers = [
                    nn.Conv2d(in_channels=hidden_size, out_channels=conv_hidden_size, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=conv_hidden_size, out_channels=1, kernel_size=1)
                ]
                self.xq_att.append(nn.Sequential(*layers))                
                self.lgd.append(nn.Linear(in_size[m], out_size))
                if att_hops > 1: 
                    layers = [
                      weight_norm(nn.Linear(in_size[m], state_size), dim=None),
                      nn.ReLU()
                    ]
                    self.q_transform.append(nn.Sequential(*layers))
        if fusioning == 'nonlinear':
            self.wx = nn.ModuleList()
            self.wq = nn.ModuleList()
            self.wxq = nn.ModuleList()
            for m in range(len(in_size)):
                if attention == 'conv' or attention == 'double_conv':
                    wx_in_size = in_size[m]
                elif attention == 'baseline':
                    wx_in_size = 2 * enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m]
                self.wx.append(nn.Linear(wx_in_size, out_size))
                self.wq.append(nn.Linear(state_size, out_size))
                self.wxq.append(nn.Linear(out_size, 1))
        elif fusioning == 'nonlinear_multiply' or fusioning == 'double_nonlinear_multiply':
            self.nonlinear = nn.ModuleList()
            for m in range(len(in_size)):
                #self.nonlinear.append(nn.ModuleList())
                if attention == 'conv' or attention == 'double_conv':
                    wx_in_size = in_size[m]
                elif attention == 'baseline':
                    wx_in_size = 2 * enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m]
                if fusioning == 'nonlinear_multiply':
                    layers = [
                        weight_norm(nn.Linear(wx_in_size, out_size), dim=None),
                        nn.ReLU()
                    ]
                elif fusioning == 'double_nonlinear_multiply':
                    layers = [
                        weight_norm(nn.Linear(wx_in_size, out_size), dim=None),
                        nn.ReLU(),
                        weight_norm(nn.Linear(out_size, out_size), dim=None),
                        nn.ReLU()
                    ]
                self.nonlinear.append(nn.Sequential(*layers))
    # Make an initial state
    def make_initial_state(self, hiddensize):
        return {name: torch.zeros(self.bsize, hiddensize, dtype=torch.float)
                for name in ('c1', 'h1')}

    # Encoder functions
    def embed_x(self, x_data, m):
        x0 = [x_data[i]
              for i in six.moves.range(len(x_data))]
        return self.emb_x[m](torch.cat(x0, 0).cuda().float())

    def forward_one_step(self, x, s, m):
        x_new = x + self.l1f_h[m](s['h1'].cuda())
        x_list = torch.split(x_new, self.enc_hsize[m], dim=1)
        x_list = list(x_list)
        c1 = torch.tanh(x_list[0]) * F.sigmoid(x_list[1]) + s['c1'].cuda() * F.sigmoid(x_list[2])
        h1 = torch.tanh(c1) * F.sigmoid(x_list[3])
        return {'c1': c1, 'h1': h1}

    def backward_one_step(self, x, s, m):
        x_new = x + self.l1b_h[m](s['h1'].cuda())
        x_list = torch.split(x_new, self.enc_hsize[m], dim=1)
        x_list = list(x_list)
        c1 = torch.tanh(x_list[0]) * F.sigmoid(x_list[1]) + s['c1'].cuda() * F.sigmoid(x_list[2])
        h1 = torch.tanh(c1) * F.sigmoid(x_list[3])
        return {'c1': c1, 'h1': h1}

    # Encoder main
    def encode(self, x):
        h1 = [None] * self.n_inputs
        for m in six.moves.range(self.n_inputs):
            if self.enc_hsize[m] > 0:
                # embedding
                seqlen = len(x[m])
                h0 = self.embed_x(x[m], m)
                # forward path
                aa = self.l1f_x[m](F.dropout(h0, training=self.train))
                fh1 = torch.split(self.l1f_x[m](F.dropout(h0, training=self.train)), self.bsize, dim=0)
                fstate = self.make_initial_state(self.enc_hsize[m])
                h1f = []
                for h in fh1:
                    fstate = self.forward_one_step(h, fstate, m)
                    h1f.append(fstate['h1'])
                # backward path
                bh1 = torch.split(self.l1b_x[m](F.dropout(h0, training=self.train)), self.bsize, dim=0)
                bstate = self.make_initial_state(self.enc_hsize[m])
                h1b = []
                for h in reversed(bh1):
                    bstate = self.backward_one_step(h, bstate, m)
                    h1b.insert(0, bstate['h1'])
                # concatenation
                h1[m] = torch.cat([torch.cat((f, b), 1)
                                   for f, b in six.moves.zip(h1f, h1b)], 0)
            else:
                # embedding only
                h1[m] = torch.tanh(self.embed_x(x[m], m))
        return h1

    # Attention
    def baseline_attention(self, h, vh, s):
        c = [None] * self.n_inputs
        for m in six.moves.range(self.n_inputs):
            bsize = self.bsize
            seqlen = int(h[m].data.shape[0] / bsize)
            csize = h[m].data.shape[1]
            asize = self.att_size

            ws = self.atW[m](s)
            vh_m = vh[m].view(seqlen, bsize, asize)
            e1 = vh_m + ws.expand_as(vh_m)
            e1 = e1.view(seqlen * bsize, asize)
            e = torch.exp(self.atw[m](torch.tanh(e1)))
            e = e.view(seqlen, bsize)
            esum = e.sum(0)
            e = e / esum.expand_as(e)
            h_m = h[m].view(seqlen, bsize, csize)
            h_m = h_m.permute(2,0,1)
            c_m = h_m * e.expand_as(h_m)
            c_m = c_m.permute(1,2,0)
            c[m] = c_m.mean(0)
        return c

    def conv_attention(self, h, s):
        x_emb = [] 
        for m in range(self.n_inputs):
            for hop in range(self.att_hops):
              h_m_permute = h[m].permute(1,0,2) 
              frames = h_m_permute.shape[1]
              x_fa = self.x_att[m](h_m_permute)
              q_fa = self.q_att[m](s)
              q_fa_expand = torch.unsqueeze(q_fa,1).expand(-1,frames,-1)
              joint_feature = x_fa * q_fa_expand
              joint_feature_reshape = torch.unsqueeze(joint_feature.permute(0,2,1),3)
              raw_attention = self.xq_att[m](joint_feature_reshape)
              raw_attention = torch.squeeze(raw_attention,3).permute(0,2,1)
              attention = F.softmax(raw_attention,dim=1).expand_as(h_m_permute)
              feature = torch.sum(attention * h_m_permute, dim=1)
              if hop == self.att_hops-1:
                  x_emb.append(feature)
              else:
                  s = self.q_transform[m](feature)
        return x_emb

    # Simple modality fusion
    def simple_modality_fusion(self, c, s):
        g = 0.
        for m in six.moves.range(self.n_inputs):
            g += self.lgd[m](F.dropout(c[m]))
        return g

    def nonlinear_modality_fusion(self, c, s):
        #pdb.set_trace()
        att_scores = [None] * self.n_inputs
        for m in range(self.n_inputs):
            att_scores[m] = self.wxq[m](F.tanh(self.wx[m](c[m]) + self.wq[m](s))).data.tolist()
        att_scores = torch.from_numpy(np.asarray(att_scores)).float().cuda()
        att_scores = F.softmax(att_scores, 0)
        g = 0.
        for m in range(self.n_inputs):
            g += att_scores[m] * self.lgd[m](F.dropout(c[m]))
        return g 

    def nonlinear_multiply_modality_fusion(self, c, s):
        for m in range(self.n_inputs):
            c[m] = self.nonlinear[m](F.dropout(c[m]))
        g = 1.
        for m in range(self.n_inputs):
            g *= c[m] 
        return g 

    # forward propagation routine
    def __call__(self, s, x, train=True):
        #pdb.set_trace() 
        self.bsize = x[0][0].shape[0]
        if self.attention == 'baseline': 
            h1 = self.encode(x)
            vh1 = [self.atV[m](h1[m]) for m in six.moves.range(self.n_inputs)]
            # attention
            c = self.baseline_attention(h1, vh1, s)
        elif self.attention == 'conv' or self.attention == 'double_conv': 
            h1 = [None] * len(self.in_size)
            for m in range(len(self.in_size)): 
                h1[m] = x[m].cuda().float()
            c = self.conv_attention(h1, s) 

        if self.fusioning == 'baseline':
            g = self.simple_modality_fusion(c, s)
        elif self.fusioning == 'nonlinear':
            g = self.nonlinear_modality_fusion(c, s)
        elif self.fusioning == 'nonlinear_multiply' or self.fusioning == 'double_nonlinear_multiply':
            g = self.nonlinear_multiply_modality_fusion(c, s)
        return torch.tanh(g)

