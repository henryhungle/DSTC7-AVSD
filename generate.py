#!/usr/bin/env python
import argparse
import logging
import math
import sys
import time
import os
import copy
import pickle
import json

import numpy as np
import six

import torch
import torch.nn as nn
import data_handler as dh
import pdb 
from pytorch_pretrained_bert import BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer, TransfoXLTokenizer

# Evaluation routine
def generate_response(model, data, batch_indices, vocab, maxlen=20, beam=5, penalty=2.0, nbest=1):
    vocablist = sorted(vocab.keys(), key=lambda s:vocab[s])
    result_dialogs = []
    model.eval()
    with torch.no_grad():
        qa_id = 0
        for dialog in data['original']['dialogs']:
            vid = dialog['image_id']
            pred_dialog = {'image_id': vid,
                           'dialog': copy.deepcopy(dialog['dialog'])}
            result_dialogs.append(pred_dialog)
            for t, qa in enumerate(dialog['dialog']):
                logging.info('%d %s_%d' % (qa_id, vid, t))
                logging.info('QS: ' + qa['question'])
                logging.info('REF: ' + qa['answer'])
                # prepare input data
                start_time = time.time()
                b = dh.make_batch(data, batch_indices[qa_id], separate_caption=train_args.separate_caption, pretrained_elmo=train_args.pretrained_elmo, pretrained_bert=train_args.pretrained_bert, bert_tokenizer=bert_tokenizer, pretrained_all=train_args.pretrained_all, bert_model=train_args.bert_model, concat_his=train_args.concat_his)
                qa_id += 1
                x = [torch.from_numpy(x) for x in b[0]]
                if train_args.concat_his:
                    h = [torch.from_numpy(h_i) for h_i in b[1]]
                else:
                    h = [[torch.from_numpy(h) for h in hb] for hb in b[1]]
                q = [torch.from_numpy(q) for q in b[2]]
                # generate sequences
                if train_args.separate_caption:
                    c = [torch.from_numpy(c) for c in b[5]]
                else:
                    c = None
                if train_args.pretrained_elmo or train_args.pretrained_bert:
                    if train_args.pretrained_all:
                        context_q, context_h, context_ai = b[-3:]
                    else:
                        context_q = b[-1]
                        context_h = None
                        context_ai = None
                else:
                    context_q = None
                    context_h = None 
                    context_ai = None
                if train_args.exclude_video:
                    x = None 
                if hasattr(train_args, "i3d_two_stream") and train_args.i3d_two_stream: 
                    min_num_frames = min(x[0].shape[0], x[1].shape[0])                    
                    x = [torch.cat([x[0][:min_num_frames,:,:], x[1][:min_num_frames,:,:]],dim=2), x[2]]
                if hasattr(model, 'generate'):
                    pred_out, _ = model.generate(x, h, q, context_q, context_h, context_ai, maxlen=maxlen, beam=beam, penalty=penalty, nbest=nbest, c=c)
                else:
                    pred_out, _ = model.module.generate(x, h, q, context_q, context_ai, maxlen=maxlen, beam=beam, penalty=penalty, nbest=nbest, c=c)
                for n in six.moves.range(min(nbest, len(pred_out))):
                    pred = pred_out[n]
                    if 'openai-gpt' in train_args.bert_model:
                        hypstr = ' '.join([vocablist[w].replace('</w>', '') for w in pred[0]])
                    elif 'gpt2' in train_args.bert_model:
                        out_tokens = []
                        for w in pred[0]:
                            token = vocablist[w]
                            if token[0].isupper():
                                out_tokens.append(token[1:])
                            else:
                                out_tokens.append(token)
                        hypstr = ' '.join(out_tokens)
                    else:
                        hypstr = ' '.join([vocablist[w] for w in pred[0]])
                    logging.info('HYP[%d]: %s  ( %f )' % (n + 1, hypstr, pred[1]))
                    if n==0:
                        pred_dialog['dialog'][t]['answer'] = hypstr
                logging.info('ElapsedTime: %f' % (time.time() - start_time))
                logging.info('-----------------------')

    return {'dialogs': result_dialogs}


##################################
# main
if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--test-path', default='', type=str,
                        help='Path to test feature files')
    parser.add_argument('--test-set', default='', type=str,
                        help='Filename of test data')
    parser.add_argument('--model-conf', default='', type=str,
                        help='Attention model to be output')
    parser.add_argument('--model', '-m', default='', type=str,
                        help='Attention model to be output')
    parser.add_argument('--maxlen', default=30, type=int,
                        help='Max-length of output sequence')
    parser.add_argument('--beam', default=3, type=int,
                        help='Beam width')
    parser.add_argument('--penalty', default=2.0, type=float,
                        help='Insertion penalty')
    parser.add_argument('--nbest', default=5, type=int,
                        help='Number of n-best hypotheses')
    parser.add_argument('--output', '-o', default='', type=str,
                        help='Output generated responses in a json file')
    parser.add_argument('--verbose', '-v', default=0, type=int,
                        help='verbose level')

    args = parser.parse_args()

    if args.verbose >= 1:
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
 
    logging.info('Loading model params from ' + args.model)
    path = args.model_conf
    with open(path, 'rb') as f:
        vocab, train_args = pickle.load(f)
    model = torch.load(args.model+'.pth.tar')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not hasattr(train_args, 'pretrained_bert'):
        train_args.pretrained_bert = False 
    if not hasattr(train_args, 'concat_his'):
        train_args.concat_his = False

    if train_args.dictmap != '':
        dictmap = json.load(open(train_args.dictmap, 'r'))
    else:
        dictmap = None
    # report data summary
    logging.info('#vocab = %d' % len(vocab))
    # prepare test data
    if train_args.pretrained_bert:
        if 'bert' in train_args.bert_model:
            bert_tokenizer = BertTokenizer.from_pretrained(train_args.bert_model)
        elif 'openai-gpt' in train_args.bert_model:
            bert_tokenizer = OpenAIGPTTokenizer.from_pretrained(train_args.bert_model)
        elif 'gpt2' in train_args.bert_model:
            bert_tokenizer = GPT2Tokenizer.from_pretrained(train_args.bert_model)
        elif 'transfo-xl' in train_args.bert_model:
            bert_tokenizer = TransfoXLTokenizer.from_pretrained(train_args.bert_model)
    else:
        bert_tokenizer = None
 
    logging.info('Loading test data from ' + args.test_set)
    test_data = dh.load(train_args.fea_type, args.test_path, args.test_set,
                        vocab=vocab, dictmap=dictmap, 
                        include_caption=train_args.include_caption, separate_caption=train_args.separate_caption,
                        pretrained_elmo=train_args.pretrained_elmo,
                        pretrained_bert=train_args.pretrained_bert, tokenizer=bert_tokenizer, bert_model=train_args.bert_model, pretrained_all=train_args.pretrained_all, concat_his=train_args.concat_his)
    test_indices, test_samples = dh.make_batch_indices(test_data, 1, separate_caption=train_args.separate_caption)
    logging.info('#test sample = %d' % test_samples)
    # generate sentences
    logging.info('-----------------------generate--------------------------')
    start_time = time.time()
    result = generate_response(model, test_data, test_indices, vocab, 
                               maxlen=args.maxlen, beam=args.beam, 
                               penalty=args.penalty, nbest=args.nbest)
    logging.info('----------------')
    logging.info('wall time = %f' % (time.time() - start_time))
    if args.output:
        logging.info('writing results to ' + args.output)
        json.dump(result, open(args.output, 'w'), indent=4)
    logging.info('done')
