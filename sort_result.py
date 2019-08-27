import numpy as np
import sys
assert len(sys.argv) == 2, 'EX: logs/val_filter_BaseCxt_layer7_bs256.txt'

lines = open(sys.argv[1]).readlines()
result = lines[:4]
lines = lines[4:]

w2ids = dict()
for i in range(0, len(lines), 16):
    keyword = lines[i].strip().split(' ')[1]
    defin = lines[i+2].strip().split('Ground Truth: ')[1]
    
    if keyword not in w2ids:
        w2ids[keyword] = {defin: [i]}
    else:
        if defin not in w2ids[keyword]:
            w2ids[keyword][defin] = [i]
        else:
            w2ids[keyword][defin].append(i)

with open(sys.argv[1], 'w') as f:
    for line in result:
        f.write(line)
    for defins in w2ids.values():
        for idList in defins.values():
            for idx in idList:
                for k in range(idx, idx+16):
                    f.write(lines[k])