## Bert2Def; PyTorch Implementation
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.1.0](https://img.shields.io/badge/pytorch-1.1.0-green.svg?style=plastic)

This repository contains the official PyTorch implementation of the following paper:

> **"What Does This Word Mean? Explaining Contextualized Embeddings with Natural Language Deﬁnition", EMNLP-IJCNLP 2019**<br>
> Ting-Yun Chang, Yun-Nung Chen<br>
> Demo website: http://140.112.29.233:5000
>
> **Abstract:** *Contextualized word embeddings have boosted many NLP tasks compared with classic word embeddings. 
However, the word with a speciﬁc sense may have different contextualized embeddings due to its various contexts. 
To further investigate what contextualized word embeddings capture, this paper analyzes whether they can indicate the
corresponding sense deﬁnitions and proposes a general framework that is capable of explaining word meanings given contextualized
word embeddings for better interpretation. The experiments show that both ELMo and BERT embeddings can be well interpreted
via a readable textual form, and the ﬁndings may beneﬁt the research community for better understanding what the embeddings capture.*

### Before Training
###### Encode the Definitions
https://tfhub.dev/google/universal-sentence-encoder-large/3

###### Extract Features from BERT
https://pypi.org/project/pytorch-pretrained-bert/

###### Extract Features from ELMo
https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md

###### Get Pretrained Static Word Embedding
https://fasttext.cc/docs/en/english-vectors.html

### Train

###### BERT-base
```bash
$ python3 main.py --model_type BERT_base --emb1_dim 768 --train_ctxVec YOUR_PATH --val_ctxVec YOUR_PATH
```

###### BERT-large
```bash
$ python3 main.py --model_type BERT_large --emb1_dim 1024 --train_ctxVec YOUR_PATH --val_ctxVec YOUR_PATH
```

###### ELMo
```bash
$ python3 main.py --model_type ELMo --emb1_dim 1024 --n_feats 3 --train_ctxVec YOUR_PATH --val_ctxVec YOUR_PATH
```

###### Baseline
```bash
$ python3 main.py --model_type baseline --emb1_dim 812 --train_ctxVec YOUR_PATH --val_ctxVec YOUR_PATH
```

### Evaluation

#### Test and view the mapping result

###### BERT-base
```bash
$ python3 main.py --test --model_type BERT_base --emb1_dim 768 --test_ctxVec YOUR_PATH --visualize
```

###### BERT-large
```bash
$ python3 main.py --test --model_type BERT_large --emb1_dim 1024 --test_ctxVec YOUR_PATH --visualize
```

###### ELMo
```bash
$ python3 main.py --test --model_type ELMo --emb1_dim 1024 --n_feats 3 --test_ctxVec YOUR_PATH --visualize
```

###### Baseline
```bash
$ python3 main.py --test --model_type baseline --emb1_dim 812 --test_ctxVec YOUR_PATH --visualize
```

##### Test Online

```bash
$ python3 online_inference.py --auto --model_type [baseline, ELMo, BERT_base, BERT_large]
```

##### Sort the result

```bash
$ python3 sort_result.py logs/YOUR_FILENAME.txt
```

##### Get Average Scores

```bash
$ python3 avg_score.py logs/YOUR_FILENAME.txt
```

##### Get BLEU/ ROUGE Scores

```bash
$ python3 get_bleu_rouge.py logs/YOUR_FILENAME.txt
```

