import numpy as np
import sys
assert len(sys.argv) == 2, "EX: logs/val_filter_BaseCxt_layer7_bs256.txt"

lines = open(sys.argv[1]).readlines()
print('[Max 100] Before Avgeraging:    {}'.format(lines[2]))
lines = lines[4:]

word2cnt = dict()
wdef2cnt = dict()
for i in range(0, len(lines), 16):
    keyword = lines[i].strip().split(' ')[1]
    gt = lines[i+2].strip().split('Ground Truth: ')[1]
    word_def = keyword+"_"+gt
    if keyword not in word2cnt:
        word2cnt[keyword] = 1
    else:
        word2cnt[keyword] += 1
    if word_def not in wdef2cnt:
        wdef2cnt[word_def] = 1
    else:
        wdef2cnt[word_def] += 1

all_word_scores = np.zeros(3)
all_wdef_scores = np.zeros(3)
num = 0
for i in range(0, len(lines), 16):
    score = np.zeros(3)
    keyword = lines[i].strip().split(' ')[1]
    result = lines[i+14].strip()
    
    if result == "In TopK":
        score[2] = 1
        gt = lines[i+2].strip().split('Ground Truth: ')[1]
        word_def = keyword+"_"+gt
        for j in range(5):
            ans = lines[i+4+j].strip().split('  ')[1]
            if ans == gt:
                if j < 1:
                    score[0] = score[1] = 1
                else:
                    score[1] = 1
                break
        all_word_scores += 100*score / word2cnt[keyword]
        all_wdef_scores += 100*score / wdef2cnt[word_def]
    num += 1

"""
Avg over (word): words with few defintions (still >= 3) have the same importance as words with many definitions
Avg over (word, definition): penalize overfitting on the (word, def) pairs occuring many times in training set
Example:
word1: def1*6, def2*5, def3*1, def4*1, def5*1, def6*1
word2: def1*4, def2*1, def3*1
"""
print('[Max 100] Avg over (word):      {{1: {}, 5: {}, 10: {}}}'.format(*np.round(all_word_scores / len(word2cnt), 2)))
print('[Max 100] Avg over (word, def): {{1: {}, 5: {}, 10: {}}}'.format(*np.round(all_wdef_scores / len(wdef2cnt), 2)))

    
