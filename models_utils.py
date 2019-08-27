import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(999)
if use_cuda:
    torch.cuda.manual_seed_all(999)

'''
def normalize_embeddings(emb, types, mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return mean.cpu() if mean is not None else None
'''


def get_mapping_accuracy(mapping, src_loader, tgt_emb, eval_few=False):
    """
    Evaluation on contextual word embedding -> definition translation.
    """
    mapping.eval()
    torch.set_grad_enabled(False)

    result = {1: 0, 5: 0, 10: 0}
    num = 0
    eval_loss, cos_dis = 0, 0
    for i, (defID, wordID, x, y) in enumerate(src_loader):

        num += len(defID)
        defID = defID.to(device)
        wordID = wordID.to(device)
        x = x.to(device)
        y = y.to(device)

        query = mapping(x, wordID)

        # Find KNN
        similarity = query.mm(tgt_emb) # calculate a batch of queries
        topIDs = similarity.topk(10, dim=1, largest=True)[1] # (BS, 10)
        defID = defID.unsqueeze(1).expand_as(topIDs) # gold
        
        # Calculate P@K
        for k in [1, 5, 10]:
            is_match = torch.sum(defID[:, :k] == topIDs[:, :k], 1).cpu().numpy() # (batch,)
            result[k] += sum(is_match)

        if not eval_few:
            eval_loss += F.mse_loss(query, y, reduction='sum')
            cos_dis += F.cosine_embedding_loss(query, y, torch.ones(y.size(0)).to(device), reduction='sum')

        elif i==2: # only evaluate on 3 batches # use when evaluating training data
            break
    
    for k in [1, 5, 10]:
        result[k] = round(result[k]*100 / num, 2)

    if not eval_few:
        result['eval_loss'] = round((eval_loss / num).item(), 3)
        result['cos_dist'] = round((cos_dis / num).item(), 3)

    mapping.train()
    torch.set_grad_enabled(True)

    return result


def visualize_knn(mapping, src_loader, tgt_emb, id2def, result, path, dump, model_type):
    """
    Visualize contextual word embedding -> definition translation.
    """
    mapping.eval()
    torch.set_grad_enabled(False)

    if dump:
        f_emb = open('mapped_embedding_{}.txt'.format(model_type), 'w')
        word_ctx = []
        embedding = np.zeros((len(src_loader.dataset), 512), dtype=np.float32)
        offset = 0

    with open(path, 'w') as f:
        f.write("########################################################\n")
        f.write("Test {}:\n".format(os.path.basename(path)))
        f.write(str(result)+"\n")
        f.write("########################################################\n")

        for defID, wordID, x, y, word, ctx, defin in src_loader: 

            defID = defID.to(device)
            wordID = wordID.to(device)
            x = x.to(device)
            y = y.to(device)

            query = mapping(x, wordID)

            if dump:
                query.div_(query.norm(2, 1, keepdim=True).expand_as(query)) # normalize
                embedding[offset: offset+len(query)] = query.cpu().numpy()
                offset += len(query)
                word_ctx.append('[{}] {}'.format(word[0], ctx[0]))

            # Find KNN
            similarity = query.mm(tgt_emb) # calculate a batch of queries
            topIDs = similarity.topk(10, dim=1, largest=True)[1] # (BS, 10)
            defID = defID.unsqueeze(1).expand_as(topIDs) # gold

            assert len(defID) == 1 # batch_size = 1
            f.write("Keyword: {}\n".format(word[0]))
            f.write("Context: {}\n".format(ctx[0]))
            f.write("Ground Truth: {}\n".format(defin[0]))
            f.write("Selected KNN:\n")
            inK = False
            for i, idx in enumerate(topIDs[0]):
                df = id2def[idx.item()]
                if df == defin[0]:
                    inK = True
                f.write('[{}]  {}\n'.format(i, df))
            res = 'In TopK\n' if inK else 'Not in TopK\n'
            f.write(res)
            f.write("----------------------------------------------------------------------------------------------------------------\n")

    if dump:
        print(offset, len(src_loader.dataset))
        embedding = np.round(embedding, 6)
        for line, vec in zip(word_ctx, embedding):
            f_emb.write('{} ; {}\n'.format(line, ' '.join(str(v) for v in vec)))
        f_emb.close()


def get_reverse_recall(voc_size, mapping, dataloader, vis=False):
    mapping.eval()
    torch.set_grad_enabled(False)

    if vis:
        index2word = np.load('data/index2word.npy', allow_pickle=True)

    recall = []
    wordID = torch.arange(voc_size).to(device)
    for i, (syn_ids, x, y, word, ctx, defin) in enumerate(tqdm(dataloader)): # an example
        x = x.expand(voc_size, *x.squeeze().shape).to(device)
        y = y.squeeze().to(device)
        syn_ids = syn_ids.squeeze(0).numpy()

        y_ = mapping(x, wordID)
        similarity = y_.mv(y) # [voc_size]
        topIDs = similarity.topk(len(syn_ids), largest=True)[1].cpu().numpy()

        true_pos = len(set(topIDs).intersection(set(syn_ids)))
        recall.append(100*true_pos/len(syn_ids))
        
        if vis:
            print('\nRecall:', np.round(recall[-1]))
            print('Target word:', word[0])
            print('Context:', ctx[0])
            print('Definition:', defin[0])
            print('Pred:', index2word[topIDs])
            print('Dict:', index2word[syn_ids])
            input()

        if i == 5000:
            break

    print('\nAvg Recall: {:.1f}%'.format(np.mean(recall)))

        

def build_model(params, pretrain):
    """
    Build all components of the model.
    """
    if params.model_type == 'baseline':
        mapping = Mapping_Base(params, pretrain).to(device)
    else:
        mapping = Mapping_BERT(params, pretrain).to(device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, mapping.parameters()) , lr=params.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=params.patience, verbose=True)
    
    if params.load or params.test:
        if params.load_epoch:
            path = os.path.join(params.ckpt_dir, 'model_{}.tar'.format(params.load_epoch))
        else:
            path = os.path.join(params.ckpt_dir, params.best_model_name)

        checkpoint = torch.load(path)
        mapping.load_state_dict(checkpoint['mapping'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load model from {} !!!'.format(path))        

    print("========================================")
    print(mapping)
    print("========================================")

    return optimizer, scheduler, mapping


def get_tuned_word_embedding(mapping, out_path="vec.txt"):
    embedding = mapping.embed.weight.cpu().detach().numpy()
    index2word = np.load('data/index2word.npy')

    with open(out_path, 'w') as f:
        for i, vec in enumerate(embedding):
            f.write('{} {}\n'.format(index2word[i], ' '.join(str(v) for v in vec)))
            

class LinearBlock(torch.nn.Module):
    def __init__(self, dim):
        super(LinearBlock, self).__init__()

        self.main = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
#            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(dim, dim),
            torch.nn.BatchNorm1d(dim))

    def forward(self, x):
        return self.main(x)


class Mapping_Base(torch.nn.Module):
    def __init__(self, params, pretrain):
        super(Mapping_Base, self).__init__()
        self.in_dim = params.emb1_dim
        self.out_dim = params.emb2_dim
        self.n_layers = params.n_layers
        self.freeze = params.freeze
        
        self.embed = torch.nn.Embedding.from_pretrained(pretrain, freeze=self.freeze)            

        layers = []
#        self.bn = torch.nn.BatchNorm1d(self.out_dim)        
        layers.append(torch.nn.Linear(self.in_dim, self.out_dim))
        for i in range(self.n_layers):
            layers.append(LinearBlock(self.out_dim))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x, w_id):
        x = torch.cat((self.embed(w_id), x), dim=1)
        out = self.mlp(x)
        
        return out
        

class Mapping_BERT(torch.nn.Module): # BERT-base, BERT-large, ELMo
    def __init__(self, params, pretrain):
        super(Mapping_BERT, self).__init__()
        self.w_dim = params.word_dim
        self.in_dim = params.emb1_dim
        self.out_dim = params.emb2_dim
        self.model_type = params.model_type
        self.n_feats = params.n_feats
        self.n_layers = params.n_layers
        self.max_pieces = 3 #params.n_BPE
        self.freeze = params.freeze

        self.embed = torch.nn.Embedding.from_pretrained(pretrain, freeze=self.freeze)
        self.weights = torch.nn.Parameter(torch.ones(self.n_feats, 1)) # init become uniform after softmax

#        self.bn = torch.nn.BatchNorm1d(self.out_dim)        
        if 'BERT' in self.model_type:
            self.n_filters = self.out_dim // 2
            self.conv1d = torch.nn.Conv1d(in_channels=self.in_dim, out_channels=self.n_filters, kernel_size=3) 
            self.conv1d2 = torch.nn.Conv1d(in_channels=self.in_dim, out_channels=self.n_filters, kernel_size=1)
            # no max_pool for self.conv1d since max_pieces = kernel_size = 3
            self.max_pool2 = torch.nn.MaxPool1d(self.max_pieces) # max_pieces - kernel +1
            self.n_layers -= 1

        layers = []
        hidden_dim = self.w_dim + self.in_dim if self.model_type == 'ELMo' else self.w_dim + self.out_dim
        layers.append(torch.nn.Linear(hidden_dim, self.out_dim)) # concat (target word embedding, fixed-sized-ctx embedding)
        for i in range(self.n_layers):
            layers.append(LinearBlock(self.out_dim))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x, w_id):
        if 'BERT' in self.model_type:
            x = x.view(-1, self.in_dim, self.max_pieces)         # reshape to parallel n_feats features
            x1 = self.conv1d(x)
            x2 = self.max_pool2(F.relu(self.conv1d2(x)))
            x1 = x1.view(-1, self.n_feats, self.n_filters)       # reshape back
            x2 = x2.view(-1, self.n_feats, self.n_filters)
            x = torch.cat((x1, x2), dim=-1)

        x = torch.sum(F.softmax(self.weights, dim=0) * x, dim=1) # weighted sum the n_feats features
        x = torch.cat((self.embed(w_id), x), dim=1)
        out = self.mlp(x)
        
        return out


class LinearBlock2(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super(LinearBlock2, self).__init__()
        self.activ = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(dim1, dim2)
        self.bn = ConditionalBatchNorm1d(dim2)

    def forward(self, x):
        out = self.activ(x)
        out = self.fc(out)
        out = self.bn(out)
        return out


class TransBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TransBlock, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, out_dim)
#        self.activ = torch.nn.ReLU(inplace=True)
#        self.fc2 = torch.nn.Linear(out_dim, out_dim)

    def forward(self, x):
        out = self.fc1(x)
#        out = self.activ(out)
#        out = self.fc2(out)
        return out
        

class Mapping2(torch.nn.Module):
    def __init__(self, params):
        super(Mapping2, self).__init__()
        self.in_dim = params.emb1_dim
        self.out_dim = params.emb2_dim
        self.n_layers = params.n_layers

        pretrain = torch.tensor(load.get_voc().embedding)
        self.embed = torch.nn.Embedding.from_pretrained(pretrain, freeze=False)            

        self.layers = []
        self.affine_generators = []
        self.layers.append(torch.nn.Linear(self.in_dim, self.out_dim))
        for _ in range(self.n_layers):
            self.layers.append(LinearBlock2(self.out_dim, self.out_dim))
            self.affine_generators.append(TransBlock(self.out_dim, self.out_dim*2))
        self.MapModels = torch.nn.Sequential(*self.layers)
        self.ConModels = torch.nn.Sequential(*self.affine_generators)
        

    def forward(self, x, w_id): 
        out = self.layers[0](torch.cat((self.embed(w_id), x), dim=1))

        for i in range(1, self.n_layers+1):
            cBN_params = self.affine_generators[i-1](x)
            self.assign_adain_params(cBN_params, self.layers[i])
            out = self.layers[i](out)
        
        return out


    def assign_adain_params(self, cBN_params, model):
        for m in model.modules():
            if m.__class__.__name__ == "ConditionalBatchNorm1d":
                m.weight = cBN_params[:, :m.num_features]
                m.bias = cBN_params[:, m.num_features:2*m.num_features]


class ConditionalBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features):
        super(ConditionalBatchNorm1d, self).__init__()
        self.num_features = num_features
        # weight and bias are dynamically assigned
        self.cbn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.weight = None
        self.bias = None
        
    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling ConditionalBatchNorm1d!"
        out = self.weight * self.cbn(x) + self.bias
        return out
