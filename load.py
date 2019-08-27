import json
import pickle
import numpy as np
import os
import glob
import torch
from torch.utils import data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_def2id_defEmbs(def_dir):
    with open(os.path.join(def_dir, 'def2id'), 'rb') as f:
        def2id = pickle.load(f)
    
    all_def_embs = np.load(os.path.join(def_dir, 'all_def_embs.npy'))
    all_def_embs = torch.tensor(all_def_embs).to(device).transpose(0, 1).contiguous() # T for (bs, 512)*(512, #)

    return def2id, all_def_embs


def get_pretrained_w2v(path, dim):
    w2v = dict()
    with open(path, 'r') as f:
        next(f) # pass the first line if needed
        for idx, line in enumerate(f):
            word, vec = line.strip().split(' ', 1)
            vec = np.fromstring(vec, sep=' ', dtype=np.float32)
            if len(vec) != dim: continue
            if word not in w2v:
                w2v[word] = vec
    print("Num pretrained word vetors:", len(w2v))

    return w2v


def get_voc(voc_path, pre_path, words_path, dim):
    try:
        voc = torch.load(voc_path)
    except FileNotFoundError:
        print("Voc not found ! Building Voc from pretrained word embedding ...")
        w2v = get_pretrained_w2v(pre_path, dim)
        voc = Voc()
        words = set(open(words_path).read().splitlines())

        for w in words:
            if w in w2v:
                voc.add_word(w, w2v[w])

        torch.save(voc, voc_path)
    print("Voc size:", voc.n_words)

    return voc


class Voc:
    def __init__(self):
        self.word2index = {}
        self.embedding = []
        self.n_words = 0

    def add_word(self, word, vec):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.embedding.append(vec)
            self.n_words += 1


# TODO:  # efficiency of _getitem_, preprocess
class myDataset(data.Dataset):

    def __init__(self, params, mode, input_file, ctx_file, def_file, def2id, voc, visualize):
        self.isVis = visualize
        self.isRev = params.reverse
        self.mode = mode
        self.model_type = params.model_type
        self.zero_shot = params.zero
        self.dataset = []

        if self.zero_shot:
            with open(params.unseen_path, 'rb') as f:
                self.unseen_voc = pickle.load(f)

        self.preprocess(input_file, ctx_file, def_file, def2id, voc, params.syn_path)

        self.num_data = len(self.dataset)


    def preprocess(self, input_file, ctx_file, def_file, def2id, voc, syn_path):
        ctx_vecs = np.load(ctx_file) # different features: context embedding, ELMo, BERT-base BERT-large
        def_vecs = np.load(def_file)
        
        print('context-dependent embedding:', ctx_vecs.shape)
        print('definition embedding:', def_vecs.shape)
        assert len(ctx_vecs) == len(def_vecs), "input error, file sizes mismatch !"

        if self.isRev:
            synonyms = open(syn_path).read().splitlines()

        oov = 0
        with open(input_file, 'r') as f:
            for i, line in enumerate(f):
                keyword, context, defin = line.split(';')
                keyword = keyword.strip()
                context = context.strip()
                defin = defin.strip()
                
                if keyword not in voc.word2index:
                    oov += 1
                    continue

                if self.zero_shot: # exclude the words in unseen set during training, and test them only
                    op = (keyword not in self.unseen_voc) if self.mode=='test' else (keyword in self.unseen_voc)
                    if op: continue

                if self.isRev:
                    syns = set([voc.word2index[w] for w in synonyms[i].split() if w in voc.word2index])
                    if len(syns) == 0: continue  # no synonyms
                    syns.add(voc.word2index[keyword])
                    self.dataset.append([-1, list(syns), ctx_vecs[i], def_vecs[i], keyword, context, defin])
                else:
                    self.dataset.append([def2id[defin], voc.word2index[keyword], ctx_vecs[i], def_vecs[i], keyword, context, defin])

        print('Num oov:', oov)


    def __getitem__(self, index):
        defID, wordID, ctx_vec, def_vec, keyword, context, defin = self.dataset[index]
        if self.isVis:
            return torch.tensor(defID), torch.tensor(wordID), torch.FloatTensor(ctx_vec), torch.FloatTensor(def_vec), keyword, context, defin
        elif self.isRev:
            return torch.tensor(wordID), torch.FloatTensor(ctx_vec), torch.FloatTensor(def_vec), keyword, context, defin
        else:
            return torch.tensor(defID), torch.tensor(wordID), torch.FloatTensor(ctx_vec), torch.FloatTensor(def_vec)

    def __len__(self):
        return self.num_data


def get_loader(params, input_file, ctx_file, def_file, def2id, voc, batch_size, mode, visualize=False):
    dataset = myDataset(params, mode, input_file, ctx_file, def_file, def2id, voc, visualize)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=(mode=='train'), drop_last=(mode=='train'))
    print("Get {} dataloader, size: {} !".format(mode, dataset.num_data))

    return dataloader
