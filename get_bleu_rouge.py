import nltk.translate.bleu_score as bleu_score
from rouge import Rouge
import sys
assert len(sys.argv) == 2, "EX: logs/val_filter_BaseCxt_layer7_bs256.txt"

lines = open(sys.argv[1]).read().splitlines()[4:]

rouge = Rouge()
myAnswers = []
references = []
for i in range(0, len(lines), 16):
    my = lines[i+4].split('  ', 1)[1]
    ref = lines[i+2].strip().split('Ground Truth: ')[1]
    myAnswers.append(my)
    references.append(ref)

#print(myAnswers[:3])
#print(references[:3])
BLEUscore = bleu_score.corpus_bleu([[r.split()] for r in references], [my.split() for my in myAnswers])

Rougescore = rouge.get_scores(myAnswers, references, avg=True)
print('[Max 100] BLEU: {:.1f}, ROUGE-L:F1 {:.1f}\n'.format(100*BLEUscore, 100*Rougescore['rouge-l']['f']))
