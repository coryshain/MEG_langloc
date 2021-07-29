import sys
import math
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import f1_score

def permutation_test(gold, preds, n=10000):
    classes, counts = np.unique(gold, return_counts=True)
    majority = classes[np.argmax(counts)]
    baseline = majority * np.ones_like(gold)
    baseline_f1 = f1_score(gold, baseline, average='macro')
    model_f1 = f1_score(gold, preds, average='macro')
    f1_diff = model_f1 - baseline_f1

    pred_table = np.stack([preds, baseline], 1)

    hits = 0

    for i in range(n):
        if i == 0 or (i + 1) % 10 == 0:
            sys.stderr.write('\r%d/%d' % (i + 1, n))
            sys.stderr.flush()
        shuffle = (np.random.random((len(pred_table))) > 0.5).astype('int')
        m1 = pred_table[np.arange(len(pred_table)),shuffle]
        m2 = pred_table[np.arange(len(pred_table)),1-shuffle]
        _f1_diff = math.fabs(f1_score(gold, m1, average='macro') - f1_score(gold, m2, average='macro'))
        if _f1_diff >= f1_diff:
            hits += 1

    p = float(hits+1)/(n+1)

    sys.stderr.write('\n')
    sys.stderr.flush()

    return p, f1_diff

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Paired permutation test of classifier F1 against a majority class baseline.
    ''')
    argparser.add_argument('path', help='Path to prediction table.')
    args = argparser.parse_args()

    df = pd.read_csv(args.path, sep=' ')

    print(permutation_test(df.CDRobs.values, df.CDRpreds.values))

