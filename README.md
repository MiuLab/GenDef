## Bert2Def; PyTorch Implementation
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.1.0](https://img.shields.io/badge/pytorch-1.1.0-green.svg?style=plastic)

This repository contains the official PyTorch implementation of the following paper:

> **"What Does This Word Mean? Explaining Contextualized Embeddings with Natural Language Deﬁnition"**, EMNLP-IJCNLP 2019
<br>
> Ting-Yun Chang, Yun-Nung Chen<br>
> Demo website: http://140.112.29.233:5000
>
> **Abstract:** *Contextualized word embeddings have boosted many NLP tasks compared with classic word embeddings. 
However, the word with a speciﬁc sense may have different contextualized embeddings due to its various contexts. 
To further investigate what contextualized word embeddings capture, this paper analyzes whether they can indicate the
corresponding sense deﬁnitions and proposes a general framework that is capable of explaining word meanings given contextualized
word embeddings for better interpretation. The experiments show that both ELMo and BERT embeddings can be well interpreted
via a readable textual form, and the ﬁndings may beneﬁt the research community for better understanding what the embeddings capture.*

### Training
```bash
$ bash run.sh
```

### Evaluation
```bash
$ bash eval.sh
```
