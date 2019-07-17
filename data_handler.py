#!/usr/bin/env python

import copy
import logging
import sys
import time
import os
import six
import pickle
import json
import numpy as np
import pdb 
import gensim 
import torch 

def get_npy_shape(filename):
    # read npy file header and return its shape
    with open(filename, 'rb') as f:
        if filename.endswith('.pkl'):
            shape = pickle.load(f).shape
        else:
            major, minor = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    return shape

def align_vocab(pretrained_vocab, vocab, pretrained_weights):
    for module, module_wt in pretrained_weights.items():
        for layer, layer_wt in module_wt.items():
            if 'embed' in layer:
                print("Aligning word emb for layer {} in module {}...".format(layer, module))
                print("Pretrained emb of shape {}".format(layer_wt.shape))
                emb_dim = layer_wt.shape[1]
                embs = np.zeros((len(vocab), emb_dim), dtype=np.float32)
                count = 0 
                for k,v in vocab.items():
                    if k in pretrained_vocab:
                        embs[v] = layer_wt[pretrained_vocab[k]]
                    else:
                        count += 1 
                pretrained_weights[module][layer] = embs
                print("Aligned emb of shape {}".format(embs.shape))
                print("Number of unmatched words {}".format(count))
    return pretrained_weights

def get_word_emb(vocab, pretrained_word_emb):
    words = set(vocab.keys())
    print("Loading pretrained word emb {}...".format(pretrained_word_emb))
    if 'GoogleNews-vectors-negative300.bin' in pretrained_word_emb:
        pretrained_emb_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_word_emb, binary=True)
        emb_dim = 300
    else:
        with open(pretrained_word_emb, "r") as f:
            lines = f.readlines()
        if len(lines[0].split())==2: lines = lines[1:]
        emb_dim = len(lines[0].split())-1
    print("Embed dim is {}".format(emb_dim))
    embs = np.zeros((len(words), emb_dim), dtype=np.float32)
    if 'GoogleNews-vectors-negative300.bin' in pretrained_word_emb:
        for word in vocab:
            idx = vocab[word]
            if word in pretrained_emb_model:
                embs[idx] = pretrained_emb_model[word]
                words.remove(word)
    else:
        for line in lines:
            tokens = line.split()
            word = tokens[0]
            if word in vocab:
                emb = [float(i) for i in tokens[1:]]
                idx = vocab[word]
                embs[idx] = np.asarray(emb)
                words.remove(word)
    print("Number of unknown words {}".format(len(words)))
    embs = np.asarray(embs)
    print("pretrained word embedding of shape {}".format(embs.shape))
    return embs 


def get_vocabulary(dataset_file, cutoff=1, include_caption=False, tokenizer=None):
    vocab = {'<unk>':0, '<sos>':1, '<eos>':2}
    dialog_data = json.load(open(dataset_file, 'r'))
    word_freq = {}
    for dialog in dialog_data['dialogs']:
        if include_caption:
            if tokenizer is not None:
                tokens = tokenizer.tokenize(dialog['caption'])
            else:
                tokens = dialog['caption'].split()
            for word in tokens:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        for key in ['question', 'answer']:
            for turn in dialog['dialog']:
                if tokenizer is not None:
                    tokens = tokenizer.tokenize(turn[key])
                else:
                    tokens = turn[key].split()
                for word in tokens:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
    for word, freq in word_freq.items():
        if freq > cutoff:
            vocab[word] = len(vocab) 
    return vocab

def words2ids(str_in, vocab, unk=0, eos=-1, tokenizer=None):
    if tokenizer is not None:
        words = tokenizer.tokenize(str_in)
    else:
        words = str_in.split()
    if eos >= 0:
        sentence = np.ndarray(len(words)+1, dtype=np.int32)
    else:
        sentence = np.ndarray(len(words), dtype=np.int32)
    for i,w in enumerate(words):
        if w in vocab:
            sentence[i] = vocab[w]
        else:
            sentence[i] = unk
    if eos >= 0:
        sentence[len(words)] = eos
    return sentence

# Load text data
def load(fea_types, fea_path, dataset_file, vocabfile='', vocab={}, include_caption=False, dictmap=None, separate_caption=False, 
    pretrained_elmo=False, pretrained_bert=False, bert_model=None, tokenizer=None,
    pretrained_all=True, concat_his=False):
    dialog_data = json.load(open(dataset_file, 'r'))
    vocablist = sorted(vocab.keys(), key=lambda s:vocab[s])
    if vocabfile != '':
        vocab_from_file = json.load(open(vocabfile,'r'))
        for w in vocab_from_file:
            if w not in vocab:
                vocab[w] = len(vocab)
    unk = vocab['<unk>']
    eos = vocab['<eos>']
    dialog_list = []
    vid_set = set()
    qa_id = 0
    for dialog in dialog_data['dialogs']:
        if include_caption:
            caption = [words2ids(dialog['caption'], vocab, eos=eos, tokenizer=tokenizer)]
        else:
            caption = [np.array([eos], dtype=np.int32)]
            if pretrained_all:
                if pretrained_elmo:
                    elmo_caption = [['<eos>']]
                if pretrained_bert:
                    if 'bert' in bert_model or 'transfo-xl' in bert_model:
                        bert_caption = ['[SEP] ']
                    elif 'gpt' in bert_model:
                        bert_caption = ['ends']

        questions = [words2ids(d['question'], vocab, tokenizer=tokenizer) for d in dialog['dialog']]
        answers = [words2ids(d['answer'], vocab, tokenizer=tokenizer) for d in dialog['dialog']]
        qa_pair = [np.concatenate((q,a,[eos])).astype(np.int32) for q,a in zip(questions, answers)]
        if pretrained_elmo:
            elmo_questions = [d['question'].split() for d in dialog['dialog']]
            if pretrained_all:
                elmo_answers = [d['answer'].split() for d in dialog['dialog']]
                elmo_qa_pairs = [ q+a+['<eos>'] for q,a in zip(elmo_questions, elmo_answers)]
        elif pretrained_bert:
            bert_questions = [d['question'] for d in dialog['dialog']]
            if pretrained_all:
                bert_answers = [d['answer'] for d in dialog['dialog']]
                if 'bert' in bert_model or 'transfo-xl' in bert_model:
                    bert_qa_pairs = [q+ ' ' + a + ' [SEP]' for q,a in zip(bert_questions, bert_answers)]
                elif 'gpt' in bert_model:
                    bert_qa_pairs = [q + ' ' + a + ' ends' for q,a in zip(bert_questions, bert_answers)]
        vid = dictmap[dialog['image_id']] if dictmap is not None else dialog['image_id']
        vid_set.add(vid)
        for n in six.moves.range(len(questions)):
            if include_caption and separate_caption:
                caption_cp = copy.copy(caption[0])
                history = [np.array([eos], dtype=np.int32)]
            else:
                history = copy.copy(caption)
                if pretrained_all:
                    if pretrained_elmo:
                        elmo_history = copy.copy(elmo_caption)
                    if pretrained_bert:
                        bert_history = copy.copy(bert_caption)
            for m in six.moves.range(n):
                history.append(qa_pair[m])
                if pretrained_all:
                    if pretrained_elmo:
                        elmo_history.append(elmo_qa_pairs[m])
                    if pretrained_bert:
                        bert_history.append(bert_qa_pairs[m])
            question = np.concatenate((questions[n], [eos])).astype(np.int32)
            answer_in = np.concatenate(([eos], answers[n])).astype(np.int32)
            answer_out = np.concatenate((answers[n], [eos])).astype(np.int32)
            if pretrained_elmo:
                elmo_question = elmo_questions[n] + ['<eos>']
                if pretrained_all: 
                    elmo_answer_in = ['<eos>'] + elmo_answers[n]
                    elmo_answer_out = elmo_answers[n] + ['<eos>']
            elif pretrained_bert:
                if 'bert' in bert_model or 'transfo-xl' in bert_model:
                    bert_question = bert_questions[n] + ' [SEP]'
                elif 'gpt' in bert_model:
                    bert_question = bert_questions[n] + ' ends'
            item = [vid, qa_id, history, question, answer_in, answer_out]
            if include_caption and separate_caption:
                item.append(caption_cp)
            if pretrained_elmo:
                item.append(elmo_question)
                if pretrained_all:
                    if concat_his:
                        concat_elmo_history = []
                        for t in elmo_history:
                            concat_elmo_history += t
                        elmo_history = concat_elmo_history
                    item.append(elmo_history)
                    item.append(elmo_answer_in)
                    item.append(elmo_answer_out)
            elif pretrained_bert:
                item.append(bert_question)
                if pretrained_all:
                    if concat_his:
                        bert_history = ' '.join(bert_history)
                    item.append(bert_history)
                    item.append(None)  #TODO: use real variables  
                    item.append(None)   
            dialog_list.append(item)
            qa_id += 1

    data = {'dialogs': dialog_list, 'vocab': vocab, 'features': [], 
            'original': dialog_data}
    for ftype in fea_types:
        basepath = fea_path.replace('<FeaType>', ftype)
        features = {}
        for vid in vid_set:
            filepath = basepath.replace('<ImageID>', vid)
            shape = get_npy_shape(filepath)
            features[vid] = (filepath, shape[0])
        data['features'].append(features)
        
    return data 

def make_batch_indices(data, batchsize=100, max_length=20, separate_caption=False):
    # Setup mini-batches
    idxlist = []
    for n, dialog in enumerate(data['dialogs']):
        vid = dialog[0]  # video ID
        x_len = []
        for feat in data['features']:
            value = feat[vid]
            size = value[1] if isinstance(value, tuple) else len(value)
            x_len.append(size)

        qa_id = dialog[1]  # QA-pair id
        h_len = len(dialog[2]) # history length
        q_len = len(dialog[3]) # question length
        a_len = len(dialog[4]) # answer length
        if separate_caption:
            c_len = len(dialog[6])
            idxlist.append((vid, qa_id, x_len, h_len, q_len, a_len, c_len))
        else:
            idxlist.append((vid, qa_id, x_len, h_len, q_len, a_len))

    if batchsize > 1:
        if separate_caption:
            idxlist = sorted(idxlist, key=lambda s:(-s[3],-s[6],-s[2][0],-s[4],-s[5]))
        else:
            idxlist = sorted(idxlist, key=lambda s:(-s[3],-s[2][0],-s[4],-s[5]))

    n_samples = len(idxlist)
    batch_indices = []
    bs = 0
    while bs < n_samples:
        in_len = idxlist[bs][3]
        bsize = int(batchsize / int(in_len / max_length + 1))
        be = min(bs + bsize, n_samples) if bsize > 0 else bs + 1
        #pdb.set_trace()
        x_len = [ max(idxlist[bs:be], key=lambda s:s[2][j])[2][j]
                for j in six.moves.range(len(x_len))]
        h_len = max(idxlist[bs:be], key=lambda s:s[3])[3]
        q_len = max(idxlist[bs:be], key=lambda s:s[4])[4]
        a_len = max(idxlist[bs:be], key=lambda s:s[5])[5]
        if separate_caption:
            c_len = max(idxlist[bs:be], key=lambda s:s[6])[6]
        vids = [ s[0] for s in idxlist[bs:be] ]
        qa_ids = [ s[1] for s in idxlist[bs:be] ]
        if separate_caption:
            batch_indices.append((vids, qa_ids, x_len, h_len, q_len, a_len, c_len, be - bs))
        else:
            batch_indices.append((vids, qa_ids, x_len, h_len, q_len, a_len, be - bs))
        bs = be
            
    return batch_indices, n_samples

def prep_context_q(context_q_batch, tokenizer):
    tokenized = [tokenizer.tokenize(t) for t in context_q_batch]
    max_len = max([len(t) for t in tokenized])
    out = []
    for t in tokenized:
        padded_text = ['[PAD]']* max_len
        padded_text[:len(t)] = t 
        out.append(tokenizer.convert_tokens_to_ids(padded_text))
    return torch.tensor(out)

def prep_context_h(context_h_batch, tokenizer):
    out = [] 
    for context_h in context_h_batch:
        out.append(prep_context_q(context_h, tokenizer))
    return out

def make_batch(data, index, eos=1, separate_caption=False, skip=[1,1,1], 
        pretrained_elmo=False, pretrained_bert=False, bert_tokenizer=None, pretrained_all=True, bert_model=None,
        concat_his=False):
    if separate_caption:
        x_len, h_len, q_len, a_len, c_len, n_seqs = index[2:]
    else:
        x_len, h_len, q_len, a_len, n_seqs = index[2:]
    feature_info = data['features']
    for j in six.moves.range(n_seqs):
        vid = index[0][j]
        fea = [np.load(fi[vid][0])[::skip[idx]] for idx,fi in enumerate(feature_info)]
        if j == 0:
            x_batch = [np.zeros((x_len[i], n_seqs, fea[i].shape[-1]),dtype=np.float32)
              if len(fea[i].shape)==2 else np.zeros((x_len[i], n_seqs, fea[i].shape[-2], fea[i].shape[-1]),dtype=np.float32)
              for i in six.moves.range(len(x_len))]

        for i in six.moves.range(len(feature_info)):
            x_batch[i][:len(fea[i]), j] = fea[i]

    empty_sentence = np.array([eos], dtype=np.int32)
    elmo_empty_sentence = ['<eos>']
    if 'bert' in bert_model or 'transfo-xl' in bert_model:
        bert_empty_sentence = '[SEP] '
    elif 'gpt' in bert_model:
        bert_empty_sentence = 'ends'

    if concat_his:
        h_batch = []
    else:
        h_batch = [ [] for _ in six.moves.range(h_len) ]
    q_batch = []
    a_batch_in = []
    a_batch_out = []
    if separate_caption:
        c_batch = [] 
    if pretrained_elmo or pretrained_bert: 
        context_q_batch = []
        if pretrained_all:
            if concat_his:
                context_h_batch = []
            else:
                context_h_batch = [[] for _ in range(h_len)]
            context_a_in_batch = []
            #context_a_out_batch = []
    dialogs = data['dialogs']
    for i in six.moves.range(n_seqs):
        qa_id = index[1][i]
        if separate_caption:
            history, question, answer_in, answer_out, caption = dialogs[qa_id][2:7]
        else:
            history, question, answer_in, answer_out = dialogs[qa_id][2:6]
        if pretrained_elmo or pretrained_bert:
            if pretrained_all:
                context_question, context_history, context_a_in, context_a_out = dialogs[qa_id][-4:]
            else:
                context_question = dialogs[qa_id][-1]
        if concat_his:
            if pretrained_all and (pretrained_elmo or pretrained_bert):
                context_h_batch.append(context_history)  
            h_batch.append(np.concatenate(history))
        else:
            for j in six.moves.range(h_len):
                if j < len(history):
                    h_batch[j].append(history[j])
                    if pretrained_all:
                        context_h_batch[j].append(context_history[j])
                else:
                    h_batch[j].append(empty_sentence)
                    if pretrained_all:
                        if pretrained_elmo:
                            context_h_batch[j].append(elmo_empty_sentence)
                        elif pretrained_bert:
                            context_h_batch[j].append(bert_empty_sentence)
        
        q_batch.append(question)
        a_batch_in.append(answer_in)
        a_batch_out.append(answer_out)
        if separate_caption:
            c_batch.append(caption)
        if pretrained_elmo or pretrained_bert:
            context_q_batch.append(context_question)
            if pretrained_all:
                context_a_in_batch.append(context_a_in)
                #context_a_out_batch.append(context_a_out)
    out = [x_batch, h_batch, q_batch, a_batch_in, a_batch_out]
    if separate_caption:
        out.append(c_batch)
    if pretrained_bert:
        context_q_batch = prep_context_q(context_q_batch, bert_tokenizer)
        if pretrained_all:
            if concat_his:
                context_h_batch = prep_context_q(context_h_batch, bert_tokenizer)
            else:
                context_h_batch = prep_context_h(context_h_batch, bert_tokenizer)
                for idx, h_b in enumerate(context_h_batch): 
                    maxlen = max([len(h) for h in h_batch[idx]])
                    if h_b.shape[-1] != maxlen: pdb.set_trace()
    if pretrained_elmo or pretrained_bert:
        out.append(context_q_batch)
        if pretrained_all:
            out.append(context_h_batch)
            out.append(context_a_in_batch)
            #out.append(context_a_out_batch)

    return out


def feature_shape(data):
    dims = []
    for features in data["features"]:
        sample_feature = list(features.values())[0]
        if isinstance(sample_feature, tuple):
	        dims.append(np.load(sample_feature[0]).shape[-1])
        else:
            dims.append(sample_feature.shape[-1])
    return dims
