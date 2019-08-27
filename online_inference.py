import torch
import argparse
import load
import os
import re
import numpy as np
import pickle


import models_utils

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_path', type=str, default="data/voc.tar", help='path to the pretrained target word embedding')
    parser.add_argument('--def2id_path', type=str, default="data/def2id", help='path to definition_to_embedding_id')
    parser.add_argument('--all_def_embs_path', type=str, default="data/all_def_embs.npy", help='path to all definition embeddings')
    parser.add_argument('--word_dim', type=int, default=300, help='dimension of the target word embedding')
    parser.add_argument('--emb1_dim', type=int, default=768, help='dimension of the context-dependent embedding')
    parser.add_argument('--emb2_dim', type=int, default=512, help='dimension of the definition embedding')
    parser.add_argument('--model_type', type=str, default='BERT_base', help='[BERT_base, BERT_large, ELMo, baseline]')
    parser.add_argument('--n_feats', type=int, default=4, help='last * layers of contextual features to be used [BERT: 4, ELMo:3]')
    parser.add_argument('--n_layers', type=int, default=7, help='number of non-linear layers')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_BERT_cnn13_base_w', help='directory to save the checkpoint')
    parser.add_argument('--best_model_name', type=str, default='best_model_out_of_100.tar', help='load only if load_epoch is not given')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='bert-base-uncased, bert-large-uncased')
    parser.add_argument('--auto', action='store_true', help='given the model_type, automatically set other arguments')

    args = parser.parse_args()
    args.freeze = True

    if args.auto:
        args = auto_set_args(args)
    inference(args)


def auto_set_args(params):
    if params.model_type == 'baseline':
        params.emb1_dim = 812
        params.ckpt_dir = 'ckpt_7_256_baseline_tune'

    elif params.model_type == 'ELMo':
        params.emb1_dim = 1024
        params.n_feats = 3
        params.ckpt_dir = 'ckpt_ELMo'

    elif params.model_type == 'BERT_base':
        params.emb1_dim = 768
        params.n_feats = 4
        params.ckpt_dir = 'ckpt_BERT_cnn13_base_w'
        params.bert_model = 'bert-base-uncased'

    elif params.model_type == 'BERT_large':
        params.emb1_dim = 1024
        params.n_feats = 4
        params.ckpt_dir = 'ckpt_BERT_cnn13_large_w'
        params.bert_model = 'bert-large-uncased'
    else:
        print("Models not defined !")

    return params


def load_model(params, pretrain):
    if params.model_type == 'baseline':
        mapping = models_utils.Mapping_Base(params, pretrain).to(device)
    else:
        mapping = models_utils.Mapping_BERT(params, pretrain).to(device)

    path = os.path.join(params.ckpt_dir, params.best_model_name)
    checkpoint = torch.load(path)
    mapping.load_state_dict(checkpoint['mapping'])
    print('Load model from {} !!!'.format(path))
    return mapping     


def load_req(params):
    voc = torch.load(params.voc_path)
    all_def_embs = torch.tensor(np.load(params.all_def_embs_path)).to(device)
    with open(params.def2id_path, 'rb') as f:
        def2id = pickle.load(f)
    id2def = {v: k for k, v in def2id.items()}
    return voc, all_def_embs, id2def


def parse_sent(s):
    s = re.sub(r"[^A-Za-z ]+", '', s.lower())
    s = re.sub(' +',' ', s).strip()
    return s


def get_input(voc):
    print("========================================================================")
    print("Input the Target Word:")
    word = input()
    try:
        assert word in voc.word2index
    except:
        print("OOV ERROR! Try to lemmatize the target word !")
        return None, None, None
    w_id = voc.word2index[word]
#    print(w_id)
    print("Input the Context:")
    ctx = input()
    ctx = parse_sent(ctx)
#    print(ctx)
    return word, [w_id], ctx


def find_varaint_word(word, ctx):
    ctx_lamma = ctx

    if word not in ctx_lamma:
        ctx_lamma = [lemmatizer.lemmatize(w, pos='n') for w in ctx]
        print(ctx_lamma)
        if word not in ctx_lamma:
            ctx_lamma = [lemmatizer.lemmatize(w, pos='v') for w in ctx]
            print(ctx_lamma)
        try:
            assert word in ctx_lamma
        except:
            print("Cannot found the target word in the context. Try to lemmatize it.")
            return None, None
            
    for j, (w_lamma, w_var) in enumerate(zip(ctx_lamma, ctx)): # find the position of the target word
        if w_lamma == word:
            return j, w_var


def answer(mapping, all_def_embs, id2def, ctx_emb, w_id):
    w_id = torch.LongTensor(w_id).to(device)
    ctx_emb = torch.FloatTensor(ctx_emb).to(device)
    query = mapping(ctx_emb, w_id).squeeze() # (512)
    similarity = all_def_embs.mv(query)
    topIDs = similarity.topk(5, largest=True)[1] # (5)
    for i, idx in enumerate(topIDs): # bs=1
        df = id2def[idx.item()]
        print('[{}] {}'.format(i, df))
    print("========================================================================")


def convert_examples_to_features(keyword, ctx, seq_length, tokenizer, vis=False):
    """Loads a data file into a list of `InputBatch`s."""

    tokens_a = tokenizer.tokenize(ctx)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    for j, token in enumerate(tokens_a):
        if "##" not in token:
            cat = token
            key_ids = [j]
        else:
            cat += token.split("##")[1]
            key_ids.append(j)
        if cat == keyword:
            break

    assert cat == keyword 
    key_ids = np.array(key_ids)+1 # [CLS]

    # tokens:  [CLS] the dog is hairy . [SEP]
    tokens = []
    tokens.append("[CLS]")
    for i, token in enumerate(tokens_a):
        tokens.append(token)
    tokens.append("[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length

    if vis:
        print("*** Example ***")
        print("tokens: %s" % " ".join([str(x) for x in tokens]))
        print("keyword: %s  keyword_ids: %s" % (keyword, np.array(tokens)[key_ids]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))

    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    input_mask = torch.LongTensor(input_mask).unsqueeze(0).to(device)

    clip = 3
    return input_ids, input_mask, key_ids[:clip]


def inference(params):
    voc, all_def_embs, id2def = load_req(params)

    mapping = load_model(params, torch.tensor(voc.embedding))
    mapping.eval()
    torch.set_grad_enabled(False)

    if params.model_type == 'baseline':
        import tensorflow_hub as hub
        import tensorflow as tf
        sent_embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        input_sent = tf.placeholder(tf.string, shape=(None))
        encoded = sent_embed(input_sent)
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sess.graph.finalize()

            while True:
                _, w_id, ctx = get_input(voc)
                if w_id == None: continue
                ctx_emb = np.round(sess.run(encoded, {input_sent: [ctx]}).astype(np.float64), 6) # (1, 512)
                answer(mapping, all_def_embs, id2def, ctx_emb, w_id)
            
    elif params.model_type == 'ELMo':
        from allennlp.commands.elmo import ElmoEmbedder
        elmo = ElmoEmbedder()

        while True:
            word, w_id, ctx = get_input(voc)
            if w_id == None: continue
            ctx = ctx.split()
            ctx_emb = elmo.embed_sentence(ctx) # (3, seq_len, 1024)

            word_pos, _ = find_varaint_word(word, ctx)
            if word_pos == None: continue

            ctx_emb = ctx_emb[:, word_pos][np.newaxis, :]
            answer(mapping, all_def_embs, id2def, ctx_emb, w_id)

    else:
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        from pytorch_pretrained_bert.modeling import BertModel

        tokenizer = BertTokenizer.from_pretrained(params.bert_model, do_lower_case=True)
        model = BertModel.from_pretrained(params.bert_model)
        model.to(device)
        model.eval()

        while True:
            word, w_id, ctx = get_input(voc)
            if w_id == None: continue

            _, word = find_varaint_word(word, ctx.split())
            if word == None: continue

            input_ids, input_mask, key_ids = convert_examples_to_features(word, ctx, 128, tokenizer)

            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            
            ctx_emb = np.zeros((params.n_feats, 3, params.emb1_dim))
            for j, ly_id in enumerate(reversed(range(-params.n_feats, 0))):  # -1, -2, -3 ...
                layer_output = all_encoder_layers[ly_id].detach().cpu().numpy().astype(np.float64).squeeze()
                ctx_emb[j] = np.round(layer_output[key_ids], 6).tolist() # (3, 768/1024)

            ctx_emb = np.transpose(ctx_emb, (0,2,1)) # (n_feats, 768/1024, 3)  
            answer(mapping, all_def_embs, id2def, ctx_emb, w_id)                


if __name__ == "__main__":
    main()