#!/usr/bin/env python
import argparse
import logging
import math
import sys
import time
import random
import os
import json

import numpy as np
import pickle
import six
import threading
import pdb 
from tqdm import tqdm 

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer, TransfoXLTokenizer
import data_handler as dh

from model.mmseq2seq_model import MMSeq2SeqModel
from model.multimodal_encoder import MMEncoder
from model.lstm_encoder import LSTMEncoder
from model.hlstm_encoder import HLSTMEncoder
from model.hlstm_decoder import HLSTMDecoder


def fetch_batch(dh, data, index, separate_caption, result):
    result.append(dh.make_batch(data, index, separate_caption=separate_caption, pretrained_elmo=args.pretrained_elmo, pretrained_bert=args.pretrained_bert, bert_tokenizer=bert_tokenizer, pretrained_all=args.pretrained_all, bert_model=args.bert_model, concat_his=args.concat_his))

# Evaluation routine
def evaluate(model, data, indices, parallel=False):
    start_time = time.time()
    eval_loss = 0.
    eval_num_words = 0
    model.eval()
    with torch.no_grad():
        # fetch the first batch
        batch = [dh.make_batch(data, indices[0], separate_caption=args.separate_caption, pretrained_elmo=args.pretrained_elmo, pretrained_bert=args.pretrained_bert, bert_tokenizer=bert_tokenizer, pretrained_all=args.pretrained_all, bert_model=args.bert_model, concat_his=args.concat_his)]
        # evaluation loop
        it = tqdm(six.moves.range(len(indices)), desc="evaluation", ncols=0)
        for j in it: 
            #if args.separate_caption:
            #    x_batch, h_batch, q_batch, a_batch_in, a_batch_out, c_batch = batch.pop()
            #else:
            #    x_batch, h_batch, q_batch, a_batch_in, a_batch_out = batch.pop()
            b = batch.pop()
            if j < len(indices)-1:
                prefetch = threading.Thread(target=fetch_batch, 
                                args=([dh, data, indices[j+1], args.separate_caption, batch]))
                prefetch.start()
            # propagate for training
            x = [torch.from_numpy(x) for x in b[0]]
            if args.concat_his:
                h = [torch.from_numpy(h_i) for h_i in b[1]]
            else:
                h = [[torch.from_numpy(h) for h in hb] for hb in b[1]]
            q = [torch.from_numpy(q) for q in b[2]]
            ai = [torch.from_numpy(ai) for ai in b[3]]
            ao = [torch.from_numpy(ao) for ao in b[4]]
            if args.separate_caption:
                c = [torch.from_numpy(c) for c in b[5]]
            else:
                c = None 
            if args.pretrained_elmo or args.pretrained_bert:
                if args.pretrained_all:
                    context_q, context_h, context_ai = b[-3:]
                else:
                    context_q = b[-1]
                    context_h = None
                    context_ai = None 
            else:
                context_q = None
                context_h = None 
                context_ai = None 
            if args.exclude_video:
                x = None

            if parallel:
                _, _, loss = model.module.loss(x, h, q, ai, ao, c, context_q, context_h, context_ai)
            else:
                _, _, loss = model.loss(x, h, q, ai, ao, c, context_q, context_h, context_ai)

            num_words = sum([len(s) for s in ao])
            eval_loss += loss.cpu().data.numpy() * num_words
            eval_num_words += num_words
            prefetch.join()
    model.train()

    wall_time = time.time() - start_time
    return math.exp(eval_loss/eval_num_words), wall_time

##################################
# main
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    # train, dev and test data
    parser.add_argument('--vocabfile', default='', type=str, 
                        help='Vocabulary file (.json)')
    parser.add_argument('--dictmap', default='', type=str, 
                        help='Dict id-map file (.json)')
    parser.add_argument('--fea-type', nargs='+', type=str, 
                        help='Image feature files (.pkl)')
    parser.add_argument('--train-path', default='', type=str,
                        help='Path to training feature files')
    parser.add_argument('--train-set', default='', type=str,
                        help='Filename of train data')
    parser.add_argument('--valid-path', default='', type=str,
                        help='Path to validation feature files')
    parser.add_argument('--valid-set', default='', type=str,
                        help='Filename of validation data')
    parser.add_argument('--include-caption', action='store_true',
                        help='Include caption in the history')
    parser.add_argument('--separate-caption', default=False, type=bool,
                        help='')
    parser.add_argument('--exclude-video', action='store_true',
                        help='')
    parser.add_argument('--pretrained-word-emb', default=None, type=str,
                        help='')
    parser.add_argument('--pretrained-weights', default=None, type=str,
                        help='')
    parser.add_argument('--pretrained-elmo', default=False, type=int, 
                        help='')
    parser.add_argument('--elmo-num-outputs', default=1, type=int,
                        help='')
    parser.add_argument('--finetune-elmo', default=False, type=int,
                        help='')
    parser.add_argument('--pretrained-bert', default=False, type=int,
                        help='')
    parser.add_argument('--bert-model', default='bert-base-uncased', type=str,
                        help='')
    parser.add_argument('--finetune-bert', default=False, type=int,
                        help='')
    parser.add_argument('--add-word-emb', default=True, type=int,
                        help='')
    parser.add_argument('--pretrained-all', default=True, type=int,
                        help='')
    parser.add_argument('--concat-his', default=False, type=int,
                        help='')
    # model parameters
    parser.add_argument('--model', '-m', default='', type=str,
                        help='Attention model to be output')
    # multimodal encoder parameters
    parser.add_argument('--enc-psize', '-p', nargs='+', type=int,
                        help='Number of projection layer units')
    parser.add_argument('--enc-hsize', '-u', nargs='+', type=int,
                        help='Number of hidden units')
    parser.add_argument('--att-size', '-a', default=100, type=int,
                        help='Number of attention layer units')
    parser.add_argument('--mout-size', default=100, type=int,
                        help='Number of output layer units')
    parser.add_argument('--mm-att', default='baseline', type=str,
                        help="") 
    parser.add_argument('--mm-fusioning', default='baseline', type=str,
                        help="")
    parser.add_argument('--mm-att-hops', default=1, type=int,
                        help='')
    parser.add_argument('--caption-mm-att', action='store_true',
                        help='')
    # input (question/caption) encoder parameters
    parser.add_argument('--embed-size', default=200, type=int, 
                        help='Word embedding size')
    parser.add_argument('--in-enc-layers', default=2, type=int,
                        help='Number of input encoder layers')
    parser.add_argument('--in-enc-hsize', default=200, type=int,
                        help='Number of input encoder hidden layer units')
    parser.add_argument('--q-att', default=None, type=str,
                        help='')
    parser.add_argument('--c-att', default=None, type=str,
                        help='')
    parser.add_argument('--rnn-type', default='lstm', type=str,
                        help='')
    parser.add_argument('--caption-states-att', default=False, type=bool,
                        help='')
    # history (QA pairs) encoder parameters
    parser.add_argument('--hist-enc-layers', nargs='+', type=int,
                        help='Number of history encoder layers')
    parser.add_argument('--hist-enc-hsize', default=200, type=int,
                        help='History embedding size')
    parser.add_argument('--hist-out-size', default=200, type=int,
                        help='History embedding size')
    parser.add_argument('--ft-fusioning', default='baseline', type=str,
                        help='Fusioning fetures between images and text')
    parser.add_argument('--caption-mm-fusion-out-size', default=-1, type=int,
                        help='')
    # response (answer) decoder parameters
    parser.add_argument('--dec-layers', default=2, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dec-psize', '-P', default=200, type=int,
                        help='Number of decoder projection layer units')
    parser.add_argument('--dec-hsize', '-d', default=200, type=int,
                        help='Number of decoder hidden layer units')
    parser.add_argument('--classifier', default='baseline', type=str,
                        help='')
    # Training conditions
    parser.add_argument('--optimizer', '-o', default='AdaDelta', type=str,
                        choices=['SGD', 'Adam', 'AdaDelta', 'RMSprop'],
                        help="optimizer")
    parser.add_argument('--rand-seed', '-s', default=1, type=int, 
                        help="seed for generating random numbers")
    parser.add_argument('--batch-size', '-b', default=20, type=int,
                        help='Batch size in training')
    parser.add_argument('--num-epochs', '-e', default=15, type=int,
                        help='Number of epochs')
    parser.add_argument('--max-length', default=20, type=int,
                        help='Maximum length for controling batch size')
    parser.add_argument('--n-batches', default=-1, type=int,
                        help='Number of batches in training')
    parser.add_argument('--weight-decay', default=0, type=float,
                        help='')
    parser.add_argument('--lr-scheduler', action='store_true',
                        help='')
    parser.add_argument('--lr', default=-1, type=float,
                        help='')
    # others
    parser.add_argument('--verbose', '-v', default=0, type=int,
                        help='verbose level')

    args = parser.parse_args()
    args.pretrained_elmo = bool(args.pretrained_elmo)
    args.finetune_elmo = bool(args.finetune_elmo)
    args.pretrained_bert = bool(args.pretrained_bert)
    args.finetune_bert = bool(args.finetune_bert)
    args.add_word_emb = bool(args.add_word_emb)
    args.pretrained_all = bool(args.pretrained_all)

    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    if args.dictmap != '':
        dictmap = json.load(open(args.dictmap, 'r'))
    else:
        dictmap = None

    if args.verbose >= 1:
        logging.basicConfig(level=logging.DEBUG, 
            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s %(levelname)s: %(message)s')

    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    
    # get vocabulary
    if args.pretrained_bert:
        if 'bert' in args.bert_model:
            bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        elif 'openai-gpt' in args.bert_model:
            bert_tokenizer = OpenAIGPTTokenizer.from_pretrained(args.bert_model)
        elif 'gpt2' in args.bert_model:
            bert_tokenizer = GPT2Tokenizer.from_pretrained(args.bert_model)
        elif 'transfo-xl' in args.bert_model:
            bert_tokenizer = TransfoXLTokenizer.from_pretrained(args.bert_model)
    else:
        bert_tokenizer = None
 
    logging.info('Extracting words from ' + args.train_set)
    vocab = dh.get_vocabulary(args.train_set, include_caption=args.include_caption, tokenizer=bert_tokenizer)

    if args.pretrained_word_emb is not None and 'none' not in args.pretrained_word_emb:
        pretrained_word_emb = dh.get_word_emb(vocab, args.pretrained_word_emb)
    else:
        pretrained_word_emb = None 
    # load data
    logging.info('Loading training data from ' + args.train_set)
    train_data = dh.load(args.fea_type, args.train_path, args.train_set, 
                         vocabfile=args.vocabfile, 
                         include_caption=args.include_caption, separate_caption=args.separate_caption,
                         vocab=vocab, dictmap=dictmap, 
                         pretrained_elmo=args.pretrained_elmo, pretrained_bert=args.pretrained_bert,
                         bert_model=args.bert_model, tokenizer=bert_tokenizer,
                         pretrained_all=args.pretrained_all, concat_his=args.concat_his)

    logging.info('Loading validation data from ' + args.valid_set)
    valid_data = dh.load(args.fea_type, args.valid_path, args.valid_set, 
                         vocabfile=args.vocabfile, 
                         include_caption=args.include_caption, separate_caption=args.separate_caption, 
                         vocab=vocab, dictmap=dictmap, 
                         pretrained_elmo=args.pretrained_elmo, pretrained_bert=args.pretrained_bert,
                         bert_model=args.bert_model, tokenizer=bert_tokenizer,
                         pretrained_all=args.pretrained_all, concat_his=args.concat_his)

    feature_dims = dh.feature_shape(train_data)
    logging.info("Detected feature dims: {}".format(feature_dims));

    # Prepare RNN model and load data
    caption_state_size = -1 
    if args.pretrained_weights:
        pretrained_weights = pickle.load(open(args.pretrained_weights, 'rb'))
        pretrained_conf = ('/').join(args.pretrained_weights.split('/')[:-1]) + '/avsd_model.conf'
        pretrained_vocab, _ = pickle.load(open(pretrained_conf, 'rb'))
        pretrained_weights = dh.align_vocab(pretrained_vocab, vocab, pretrained_weights)
    else:
        pretrained_weights = None 
    if args.q_att:
        in_size_decoder = args.mout_size + args.hist_out_size + args.in_enc_hsize*2
        state_size = args.in_enc_hsize*2
    else:
        in_size_decoder = args.mout_size + args.hist_out_size + args.in_enc_hsize
        state_size = args.in_enc_hsize 
    if args.separate_caption:
        if args.c_att == 'conv_sum':
            caption_state_size = args.in_enc_hsize*2
            if not args.caption_states_att:
                in_size_decoder += caption_state_size
        else:
            caption_state_size = args.in_enc_hsize
            if not args.caption_states_att:
                in_size_decoder += caption_state_size
    if args.exclude_video:
        mm_encoder = None
        in_size_decoder -= args.mout_size
    else:
        if args.caption_mm_att:
            mm_state_size = caption_state_size
        else:
            mm_state_size = state_size
        mm_encoder = MMEncoder(feature_dims, args.mout_size, enc_psize=args.enc_psize,
                          enc_hsize=args.enc_hsize, att_size=args.att_size,
                          state_size=mm_state_size, attention=args.mm_att, fusioning=args.mm_fusioning, 
                          att_hops=args.mm_att_hops)
    if args.ft_fusioning == 'caption_mm_nonlinear_multiply':
        in_size_decoder = in_size_decoder - args.mout_size - caption_state_size + args.caption_mm_fusion_out_size
    weights_init=pretrained_weights['history_encoder'] if pretrained_weights is not None else None 
    hlstm_encoder = HLSTMEncoder(args.hist_enc_layers[0], args.hist_enc_layers[1],
                          len(vocab), args.hist_out_size, args.embed_size,
                          args.hist_enc_hsize, rnn_type=args.rnn_type, embedding_init=pretrained_word_emb, weights_init=weights_init,
                          elmo_init=args.pretrained_elmo, elmo_num_outputs=args.elmo_num_outputs, finetune_elmo=args.finetune_elmo,
                          bert_init=args.pretrained_bert, bert_model=args.bert_model, finetune_bert=args.finetune_bert,
                          add_word_emb=args.add_word_emb, pretrained_all=args.pretrained_all,
                          concat_his=args.concat_his)
    weights_init=pretrained_weights['input_encoder'] if pretrained_weights is not None else None 
    input_encoder = LSTMEncoder(args.in_enc_layers, len(vocab), args.in_enc_hsize,
                          args.embed_size, attention=args.q_att, rnn_type=args.rnn_type, embedding_init=pretrained_word_emb, weights_init=weights_init, 
                          elmo_init=args.pretrained_elmo, elmo_num_outputs=args.elmo_num_outputs, finetune_elmo=args.finetune_elmo,
                          bert_init=args.pretrained_bert,  bert_model=args.bert_model, finetune_bert=args.finetune_bert,
                          add_word_emb=args.add_word_emb)
    weights_init=pretrained_weights['response_decoder'] if pretrained_weights is not None else None 
    hlstm_decoder = HLSTMDecoder(args.dec_layers, len(vocab), len(vocab), args.embed_size,
                            in_size_decoder,
                            args.dec_hsize, args.dec_psize,
                            independent=True, rnn_type=args.rnn_type,
                            classifier=args.classifier, states_att=args.caption_states_att, state_size=caption_state_size, embedding_init=pretrained_word_emb, weights_init=weights_init,
                            elmo_init=args.pretrained_elmo, elmo_num_outputs=args.elmo_num_outputs, finetune_elmo=args.finetune_elmo,
                            bert_init=args.pretrained_bert, bert_model=args.bert_model, finetune_bert=args.finetune_bert,
                            add_word_emb=args.add_word_emb, pretrained_all=args.pretrained_all)
    if args.separate_caption:
        weights_init=pretrained_weights['caption_encoder'] if pretrained_weights is not None else None 
        caption_encoder = LSTMEncoder(args.in_enc_layers, len(vocab), args.in_enc_hsize,
                          args.embed_size, attention=args.c_att, rnn_type=args.rnn_type, q_size=state_size, weights_init=weights_init)
    else:
        caption_encoder = None 
    model = MMSeq2SeqModel(mm_encoder, hlstm_encoder, input_encoder, hlstm_decoder, fusioning=args.ft_fusioning, caption_encoder=caption_encoder, 
        caption_states_att = args.caption_states_att, caption_mm_att = args.caption_mm_att, c_in_size=caption_state_size, mm_in_size=args.mout_size, out_size=args.caption_mm_fusion_out_size)

    # report data summary
    logging.info('#vocab = %d' % len(vocab))
    # make batchset for training
    logging.info('Making mini batches for training data')
    train_indices, train_samples = dh.make_batch_indices(train_data, args.batch_size,
                                                         max_length=args.max_length, separate_caption=args.separate_caption)
    logging.info('#train sample = %d' % train_samples)
    logging.info('#train batch = %d' % len(train_indices))
    # make batchset for validation
    logging.info('Making mini batches for validation data')
    valid_indices, valid_samples = dh.make_batch_indices(valid_data, args.batch_size,
                                                     max_length=args.max_length, separate_caption=args.separate_caption)
    logging.info('#validation sample = %d' % valid_samples)
    logging.info('#validation batch = %d' % len(valid_indices))
    # copy model to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    parallel = False
    path = args.model + '.conf'
    with open(path, 'wb') as f:
        pickle.dump((vocab, args), f, -1)
    path2 = args.model + '_params.txt'
    with open(path2, "w") as f: 
        for arg in vars(args):
            f.write("{}={}\n".format(arg, getattr(args, arg)))
 
    # start training 
    logging.info('----------------')
    logging.info('Start training')
    logging.info('----------------')
    # Setup optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    elif args.optimizer == 'AdaDelta':
        optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), weight_decay=args.weight_decay)

    if args.lr > 0: 
        for g in optim.param_groups:
            g['lr'] = args.lr 

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # initialize status parameters
    modelext = '.pth.tar'
    cur_loss = 0.
    cur_num_words = 0
    epoch = 0
    start_at = time.time()
    cur_at = start_at
    min_valid_ppl = 1.0e+10
    n = 0
    report_interval = int(100/args.batch_size)
    bestmodel_num = 0
    random.shuffle(train_indices)

    trace_log_path = args.model+'_trace.csv'
    with open(trace_log_path, "w") as f:
        f.write('epoch,split,perplexity\n') 
    train_log_path = args.model+'_train.csv'
    with open(train_log_path, "w") as f:  
        f.write('epoch,step,perplexity\n') 
    print("Saving training results to {}".format(train_log_path))
    print("Saving val results to {}".format(trace_log_path))

    # do training iterations
    for i in six.moves.range(args.num_epochs):
        if args.lr_scheduler:
            scheduler.step()
        logging.info('-------------------------Epoch %d : %s-----------------------' % (i+1, args.optimizer))
        train_loss = 0.
        train_num_words = 0
        # fetch the first batch
        batch = [dh.make_batch(train_data, train_indices[0], separate_caption=args.separate_caption, pretrained_elmo=args.pretrained_elmo, pretrained_bert=args.pretrained_bert, bert_tokenizer=bert_tokenizer, 
                pretrained_all=args.pretrained_all, bert_model=args.bert_model,
                concat_his=args.concat_his)]
        # train iterations
        if args.n_batches > 0: 
            n_batches = args.n_batches
        else:
            n_batches = len(train_indices)
        it = tqdm(six.moves.range(n_batches), desc="epoch {}/{}".format(i, args.num_epochs), ncols=0)
        for j in it:
            b = batch.pop()
            # fetch the next batch in parallel
            if j < len(train_indices)-1:
                prefetch = threading.Thread(target=fetch_batch, 
                                args=([dh, train_data, train_indices[j+1], args.separate_caption, batch]))
                prefetch.start()
            
            # propagate for training
            x = [torch.from_numpy(x) for x in b[0]]
            if args.concat_his:
                h = [torch.from_numpy(h_i) for h_i in b[1]]
            else:
                h = [[torch.from_numpy(h) for h in hb] for hb in b[1]]
            q = [torch.from_numpy(q) for q in b[2]]
            ai = [torch.from_numpy(ai) for ai in b[3]]
            ao = [torch.from_numpy(ao) for ao in b[4]]
            if args.separate_caption:
                c = [torch.from_numpy(c) for c in b[5]]
            else:
                c = None
            if args.pretrained_elmo or args.pretrained_bert:
                if args.pretrained_all:
                    context_q, context_h, context_ai = b[-3:]
                else:
                    context_q = b[-1]
                    context_h = None
                    context_ai = None
            else:
                context_q = None
                context_h = None
                context_ai = None 
            if args.exclude_video:
                x = None 
            if parallel:
                _, _, loss = model.module.loss(x, h, q, ai, ao, c, context_q, context_h, context_ai)
            else:
                _, _, loss = model.loss(x, h, q, ai, ao, c, context_q, context_h, context_ai)

            num_words = sum([len(s) for s in ao])
            batch_loss = loss.cpu().data.numpy()
            train_loss += batch_loss * num_words
            train_num_words += num_words

            cur_loss += batch_loss * num_words
            cur_num_words += num_words
            if (n + 1) % report_interval == 0:
                now = time.time()
                throuput = report_interval / (now - cur_at)
                perp = math.exp(cur_loss / cur_num_words)
                it.set_postfix(train_perplexity='{:.3f}'.format(perp))
                with open(train_log_path, "a") as f:
                   f.write("{},{},{:e}\n".format(i+1,n+1,perp))
                cur_at = now
                cur_loss = 0.
                cur_num_words = 0
            n += 1
            # Run truncated BPTT
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # wait prefetch completion
            prefetch.join()

        train_ppl =  math.exp(train_loss/train_num_words)
        logging.info("epoch: %d  train perplexity: %f" % (i+1, train_ppl))

        # validation step 
        logging.info('-------validation--------')
        now = time.time()
        valid_ppl, valid_time = evaluate(model, valid_data, valid_indices, parallel)
        logging.info('validation perplexity: %.4f' % (valid_ppl))
        with open(trace_log_path,"a") as f:
            f.write("{},train,{:e}\n".format(i+1,train_ppl))
            f.write("{},val,{:e}\n".format(i+1,valid_ppl))        

        # update the model via comparing with the lowest perplexity  
        modelfile = args.model + '_' + str(i + 1) + modelext
        logging.info('writing model params to ' + modelfile)
        torch.save(model, modelfile)

        if min_valid_ppl > valid_ppl:
            bestmodel_num = i+1
            logging.info('validation perplexity reduced %.4f -> %.4f' % (min_valid_ppl, valid_ppl))
            min_valid_ppl = valid_ppl
            logging.info('a symbolic link is made as ' + args.model + '_best' + modelext)
            if os.path.exists(args.model + '_best' + modelext):
                os.remove(args.model + '_best' + modelext)
            os.symlink(os.path.basename(args.model + '_' + str(bestmodel_num) + modelext), args.model + '_best' + modelext)

        cur_at += time.time() - now  # skip time of evaluation and file I/O
        logging.info('----------------')

    # make a symlink to the best model
    logging.info('the best model is epoch %d.' % bestmodel_num)
