# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 
from torch.nn.utils.weight_norm import weight_norm

torch.manual_seed(1)


class MMSeq2SeqModel(nn.Module):

    def __init__(self, mm_encoder, history_encoder, input_encoder, response_decoder, fusioning="baseline", caption_encoder=None, caption_states_att=False, caption_mm_att=False, c_in_size=-1, mm_in_size=-1, out_size=-1):
        """ Define model structure
            Args:
                history_encoder (~chainer.Chain): history encoder network
                input_encoder (~chainer.Chain): input encoder network
                response_decoder (~chainer.Chain): response decoder network
        """
        super(MMSeq2SeqModel, self).__init__()
        self.history_encoder = history_encoder
        self.mm_encoder = mm_encoder
        self.input_encoder = input_encoder
        self.response_decoder = response_decoder
        self.fusioning = fusioning
        if caption_encoder:
            self.caption_encoder = caption_encoder 
        self.caption_states_att = caption_states_att  
        self.caption_mm_att = caption_mm_att
        if fusioning == 'caption_mm_nonlinear_multiply':
            layers = [
              weight_norm(nn.Linear(c_in_size, out_size), dim=None),
              nn.ReLU()
            ]
            self.cW = nn.Sequential(*layers)
            layers = [
              weight_norm(nn.Linear(mm_in_size, out_size), dim=None),
              nn.ReLU()
            ]
            self.mmW = nn.Sequential(*layers)

    def loss(self, mx, hx, x, y, t, c=None, context_x=None, context_hx=None, context_ai=None):
        """ Forward propagation and loss calculation
            Args:
                es (pair of ~chainer.Variable): encoder state
                x (list of ~chainer.Variable): list of input sequences
                y (list of ~chainer.Variable): list of output sequences
                t (list of ~chainer.Variable): list of target sequences
                                   if t is None, it returns only states
            Return:
                es (pair of ~chainer.Variable(s)): encoder state
                ds (pair of ~chainer.Variable(s)): decoder state
                loss (~chainer.Variable) : cross-entropy loss
        """
        # encode
        # encode question x  
        ei = self.input_encoder(None, x, context_x=context_x)
        # encode history hx 
        eh = self.history_encoder(None, hx, context_x=context_hx)
        # encode caption c 
        if c is not None: 
            ec = self.caption_encoder(None, c, states_att=self.caption_states_att, q=ei) 
        if mx is not None:
            if self.caption_mm_att:
              ems = self.mm_encoder(ei, mx, ec)
            else:
              ems = self.mm_encoder(ei, mx)
        # concatenate encodings
        if mx is not None:
          features = [ei, ems, eh[-1]]
        else:
          features = [ei, eh[-1]]
        if self.fusioning == 'baseline':
            if c is not None and not self.caption_states_att: 
                es = torch.cat(features + [ec], dim=1)
            else:
                es = torch.cat(features, dim=1)
        elif self.fusioning == 'nonlinear':
            ems = (ems * F.sigmoid(ei) + ems * F.sigmoid(eh[-1]))/2
            es = torch.cat((ei, ems, eh[-1]), dim=1)
        elif self.fusioning == 'caption_mm_nonlinear_multiply':
            ec_nonlinear = self.cW(ec)
            ems_nonlinear = self.mmW(ems)
            ec_ems_joint_feature = ec_nonlinear * ems_nonlinear
            es = torch.cat((ei, eh[-1], ec_ems_joint_feature), dim=1)
        
        if hasattr(self.response_decoder, 'context_to_state') \
            and self.response_decoder.context_to_state==True:
            if c is not None and self.caption_states_att:
                ds, dy = self.response_decoder(es, None, y, ec)
            else:
                ds, dy = self.response_decoder(es, None, y)
        else:
            # decode
            if c is not None and self.caption_states_att:
                ds, dy = self.response_decoder(None, es, y, ec)
            else:
                ds, dy = self.response_decoder(None, es, y, context_y=context_ai)

        # compute loss
        if t is not None:
            tt = torch.cat(t, dim=0)
            loss = F.cross_entropy(dy, torch.tensor(tt, dtype=torch.long).cuda())
            max_index = dy.max(dim=1)[1]
            hit = (max_index == torch.tensor(tt, dtype=torch.long).cuda()).sum()
            return None, ds, loss
        else:  # if target is None, it only returns states
            return None, ds

    def generate(self, mx, hx, x, context_x, context_hx, context_ai, sos=2, eos=2, unk=0, minlen=1, maxlen=100, beam=5, penalty=1.0, nbest=1, c=None):
        """ Generate sequence using beam search
            Args:
                es (pair of ~chainer.Variable(s)): encoder state
                x (list of ~chainer.Variable): list of input sequences
                sos (int): id number of start-of-sentence label
                eos (int): id number of end-of-sentence label
                unk (int): id number of unknown-word label
                maxlen (int): list of target sequences
                beam (int): list of target sequences
                penalty (float): penalty added to log probabilities
                                 of each output label.
                nbest (int): number of n-best hypotheses to be output
            Return:
                list of tuples (hyp, score): n-best hypothesis list
                 - hyp (list): generated word Id sequence
                 - score (float): hypothesis score
                pair of ~chainer.Variable(s)): decoder state of best hypothesis
        """
        ei = self.input_encoder(None, x, context_x=context_x)
        eh = self.history_encoder(None, hx, context_x=context_hx)
        if c is not None: 
            ec = self.caption_encoder(None, c, states_att=self.caption_states_att, q=ei)
        if mx is not None:
            if self.caption_mm_att:
                ems = self.mm_encoder(ei, mx, ec)
            else:
                ems = self.mm_encoder(ei, mx)
        # concatenate encodings
        if mx is not None:
            features = [ei, ems, eh[-1]]
        else:
            features = [ei, eh[-1]]
        if self.fusioning == 'baseline':
            if c is not None and not self.caption_states_att:
                es = torch.cat(features + [ec], dim=1)
            else:
                es = torch.cat(features, dim=1)
        elif self.fusioning == 'nonlinear':
            ems = (ems * F.sigmoid(ei) + ems * F.sigmoid(eh[-1]))/2
            es = torch.cat((ei, ems, eh[-1]), dim=1)
        elif self.fusioning == 'caption_mm_nonlinear_multiply':
            ec_nonlinear = self.cW(ec)
            ems_nonlinear = self.mmW(ems)
            ec_ems_joint_feature = ec_nonlinear * ems_nonlinear
            es = torch.cat((ei, eh[-1], ec_ems_joint_feature), dim=1)

        # beam search
        ds = self.response_decoder.initialize(None, es, torch.from_numpy(np.asarray([sos])).cuda())
        hyplist = [([], 0., ds)]
        best_state = None
        comp_hyplist = []
        for l in six.moves.range(maxlen):
            new_hyplist = []
            argmin = 0
            for out, lp, st in hyplist:
                if self.caption_states_att:
                    logp = self.response_decoder.predict(st, ec)
                else:
                    logp = self.response_decoder.predict(st)
                lp_vec = logp.cpu().data.numpy() + lp
                lp_vec = np.squeeze(lp_vec)
                if l >= minlen:
                    new_lp = lp_vec[eos] + penalty * (len(out) + 1)
                    new_st = self.response_decoder.update(st, torch.from_numpy(np.asarray([eos])).cuda())
                    comp_hyplist.append((out, new_lp))
                    if best_state is None or best_state[0] < new_lp:
                        best_state = (new_lp, new_st)

                for o in np.argsort(lp_vec)[::-1]:
                    if o == unk or o == eos:  # exclude <unk> and <eos>
                        continue
                    new_lp = lp_vec[o]
                    if len(new_hyplist) == beam:
                        if new_hyplist[argmin][1] < new_lp:
                            new_st = self.response_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
                            new_hyplist[argmin] = (out + [o], new_lp, new_st)
                            argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                        else:
                            break
                    else:
                        new_st = self.response_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
                        new_hyplist.append((out + [o], new_lp, new_st))
                        if len(new_hyplist) == beam:
                            argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]

            hyplist = new_hyplist

        if len(comp_hyplist) > 0:
            maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
            return maxhyps, best_state[1]
        else:
            return [([], 0)], None
