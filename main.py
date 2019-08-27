import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
import models_utils
import load


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(999)
if use_cuda:
    torch.cuda.manual_seed_all(999)
np.random.seed(999)


def train(params):

    def2id, all_def_embs = load.get_def2id_defEmbs(params.def_dir)
    voc = load.get_voc(params.voc_path, params.w2v_path, params.words_path, params.word_dim)
    train_loader = load.get_loader(params, params.train_pair, params.train_ctxVec, params.train_defVec, def2id, voc, params.batch_size, 'train')
    val_loader = load.get_loader(params, params.val_pair, params.val_ctxVec, params.val_defVec, def2id, voc, 4096, 'val')
    del def2id

    optimizer, scheduler, mapping = models_utils.build_model(params, torch.tensor(voc.embedding))

    # Start Training
    best_score = 0 # max precision = 100
    start_e = params.load_epoch + 1 if params.load_epoch else 1
    for epoch in tqdm(range(start_e, params.max_epoch+1)):
        for i, (_, wordID, x, y) in enumerate(train_loader):

            optimizer.zero_grad()
            x = x.to(device) # [bs, num_layers, 768]
            y = y.to(device) # [bs, 512]
            wordID = wordID.to(device)

            y_ = mapping(x, wordID)
            loss = F.mse_loss(y_, y, reduction='sum') / params.batch_size
            
            loss.backward()
            optimizer.step()

        # Run evaluation to decide whether to decay lr / update best model
        result = models_utils.get_mapping_accuracy(mapping, val_loader, all_def_embs)
        scheduler.step(result['eval_loss'])

        score = result[5] # use P@5 as criterion
        if score > best_score:
            best_res = {
                'epoch': epoch,
                'batch_size': params.batch_size,
                'lr': params.lr,
                'val_loss': result['eval_loss'],
                'cos_dist': result['cos_dist'],
                'P@1': result[1].item(),
                'P@5': result[5].item(),
                'P@10': result[10].item(),
                'mapping': mapping.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(best_res, os.path.join(params.ckpt_dir, 'best_model_out_of_{}.tar'.format(params.max_epoch)))
            best_score = score
            print("[{:2d}] Save Best P@K [max 100%] :".format(epoch))
            print("[Validation Set] :")
            print(result)
            print("========================================================")


        if (epoch % params.print_epoch == 0) or (epoch % params.save_epoch == 0): 
            print('[{:2d}] loss: {:.2f}'.format(epoch, loss))
            print("P@K [max 100%] :")
            print("[Validation Set] :")
            print(result)
            print("[Training Set (average over 3 batches only)] :")
            tr_res = models_utils.get_mapping_accuracy(mapping, train_loader, all_def_embs, eval_few=True)
            print(tr_res)
                
            if epoch % params.save_epoch == 0:
                save_res = {
                    'epoch': epoch,
                    'batch_size': params.batch_size,
                    'lr': params.lr,
                    'val_loss': result['eval_loss'],
                    'P@1': result[1].item(),
                    'P@5': result[5].item(),
                    'P@10': result[10].item(),
                    'mapping': mapping.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(save_res, os.path.join(params.ckpt_dir, 'model_{}.tar'.format(epoch)))
            print("========================================================")

        
    print("########################################################")
    print("Best Result Overall:")
    print('  epoch:', best_res['epoch'])
    print('  val_loss:', best_res['val_loss'])
    print('  cos_dist:', best_res['cos_dist'])
    print('  P@1:', best_res['P@1'])
    print('  P@5:', best_res['P@5'])
    print('  P@10:', best_res['P@10'])
    print("########################################################")
    if params.model_type != 'baseline':
        print("Learned weights for different BERT layer:")
        print(F.softmax(mapping.weights, dim=0))

        

def test(params):

    def2id, all_def_embs = load.get_def2id_defEmbs(params.def_dir)
    voc = load.get_voc(params.voc_path, params.w2v_path, params.words_path, params.word_dim)

    test_loader = load.get_loader(params, params.test_pair, params.test_ctxVec, params.test_defVec, def2id, voc, 4096, 'test')
    
    _, _, mapping = models_utils.build_model(params, torch.tensor(voc.embedding))
    result = models_utils.get_mapping_accuracy(mapping, test_loader, all_def_embs)
    print("########################################################")
    print("Test {}:".format(params.test_pair))
    print(result)
    print("########################################################")

    if params.visualize:
        test_name = os.path.basename(params.test_pair).split('.')[0]
        out_log_path = os.path.join(params.log_dir, '{}_{}_layer{}_bs{}.txt'.format(test_name, params.model_type, params.n_layers, params.batch_size))
        print("out_log_path:", out_log_path)
        vis_loader = load.get_loader(params, params.test_pair, params.test_ctxVec, params.test_defVec, def2id, voc, 1, 'test', visualize=True)
        id2def = {v: k for k, v in def2id.items()}
        models_utils.visualize_knn(mapping, vis_loader, all_def_embs, id2def, result, out_log_path, params.dump, params.model_type)


def test_rev(params):
    voc = load.get_voc(params.voc_path, params.w2v_path, params.words_path, params.word_dim)
    # test reverse mapping on the train set
    test_loader = load.get_loader(params, params.train_pair, params.train_ctxVec, params.train_defVec, None, voc, 1, 'train')
    
    _, _, mapping = models_utils.build_model(params, torch.tensor(voc.embedding))

    models_utils.get_reverse_recall(voc.n_words, mapping, test_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--def_dir', type=str, default="data", help='the directory for def2id and def_embeddings')
    parser.add_argument('--train_pair', type=str, default="data/train.txt", help='.npy input file for training')
    parser.add_argument('--val_pair', type=str, default="data/val_filter.txt", help='.npy input file for validation')
    parser.add_argument('--test_pair', type=str, default="data/test_easy.txt", help='.npy input file for inference')
    parser.add_argument('--train_ctxVec', type=str, default="features/train.npy", help='.npy context-dependent embedding for training')
    parser.add_argument('--val_ctxVec', type=str, default="features/val_filter.npy", help='.npy context-dependent embedding for validation')
    parser.add_argument('--test_ctxVec', type=str, default="features/test_easy.npy", help='.npy context-dependent embedding for inference')
    parser.add_argument('--train_defVec', type=str, default="data/def_train.npy", help='.npy definition embedding for training')
    parser.add_argument('--val_defVec', type=str, default="data/def_val_filter.npy", help='.npy definition embedding for validation')
    parser.add_argument('--test_defVec', type=str, default="data/def_test_easy.npy", help='.npy definition embedding for inference')
    parser.add_argument('--syn_path', type=str, default="data/train_syn.txt", help='path to synonyms')
    parser.add_argument('--w2v_path', type=str, default="/media/tera/DATA/myData/W2V/wiki-news-300d-1M.vec", help='path to the pretrained word embedding')
    parser.add_argument('--voc_path', type=str, default="data/voc.tar", help='path to the pretrained target word embedding')
    parser.add_argument('--words_path', type=str, default="parsed/train_Voc.txt", help='path to the file containing all target words')
    parser.add_argument('--unseen_path', type=str, default="data/unseen_words", help='path to the file containing unseen target words to be held out')
    parser.add_argument('--word_dim', type=int, default=300, help='dimension of the target word embedding')
    parser.add_argument('--emb1_dim', type=int, default=768, help='dimension of the contextual keyword vector')
    parser.add_argument('--emb2_dim', type=int, default=512, help='dimension of the definition vector')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--model_type', type=str, default='BERT_base', help='[BERT_base, BERT_large, ELMo, baseline]')
    parser.add_argument('--n_feats', type=int, default=4, help='last * layers of contextual features to be used [BERT: 4, ELMo:3]')
#    parser.add_argument('--n_BPE', type=int, default=3, help='maximum word pieces to be used')
    parser.add_argument('--n_layers', type=int, default=7, help='number of non-linear layers')
    parser.add_argument('--patience', type=int, default=5, help='number of epochs with no improvement after which lr will be reduced')
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch to train the model')
    parser.add_argument('--save_epoch', type=int, default=50, help='save the model every * epochs')
    parser.add_argument('--print_epoch', type=int, default=10, help='print the result every * epochs')
    parser.add_argument('--lr', type=float, default=5*1e-4, help='learning rate')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', help='directory to save the checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to save the logs')
    parser.add_argument('--load_epoch', type=int, help='load the model pretrained for * epoches, load the best model if not given')
    parser.add_argument('--best_model_name', type=str, default='best_model_out_of_100.tar', help='load only if load_epoch is not given')
    parser.add_argument('--norm', action='store_true', default=False, help='normalize embeddings to unit vector before training')
    parser.add_argument('--load', action='store_true', default=False, help='load pretrained')
    parser.add_argument('--test', action='store_true', default=False, help='testing mode')
    parser.add_argument('--reverse', action='store_true', default=False, help='reverse mapping')
    parser.add_argument('--zero', action='store_true', default=False, help='zero-shot, input is (unseen word, unseen context)')
    parser.add_argument('--freeze', action='store_true', default=False, help='freeze the word embedding as pretrained word embedding')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize the mapping result when inference')
    parser.add_argument('--dump', action='store_true', default=False, help='dump the mapped embeddings')


    args = parser.parse_args()

    if not os.path.exists(args.ckpt_dir):
        assert not args.test, "directory args.ckpt_dir for pretrained model doesn't exist !"
        os.makedirs(args.ckpt_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.load and not args.load_epoch:
        assert args.best_model_name, "No pre-trained model is given !"
    
    assert args.model_type in ['BERT_base', 'BERT_large', 'ELMo', 'baseline'], "model_type not found !"
    print("freeze:", args.freeze, "zero-shot:", args.zero)

    if args.test:
        print("[Test Mode]")
        if args.reverse:
            test_rev(args)
        else:
            test(args)
    else:
        print("[Train Mode]")
        train(args)

if __name__ == "__main__":
    main()