import numpy as np
import pandas as pd

#load confusion matrix (exclude labels)
cm = pd.read_csv('confusion_matrix.csv', header=0).to_numpy()[:,1:].astype(float)

#true positives
tp = np.diag(cm)
#false positive
fp = cm.sum(axis=0) - tp
#false negatives
fn = cm.sum(axis=1) - tp

#Calculate precision, recall, and F1
precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
f1_score  = np.divide(2 * precision * recall, precision + recall,
                      out=np.zeros_like(precision, dtype=float),
                      where=(precision + recall) != 0)

#turn into dataframe
df = pd.DataFrame({
    'precision': precision,
    'recall':    recall,
    'f1-score':  f1_score
}, index=[f'Class {i}' for i in range(len(tp))])

#Average macro and micro
macro_f1 = f1_score.mean()
micro_precision = tp.sum() / (tp.sum() + fp.sum())
micro_recall    = tp.sum() / (tp.sum() + fn.sum())
micro_f1        = (2 * micro_precision * micro_recall
                   / (micro_precision + micro_recall))

df.loc['macro'] = [precision.mean(), recall.mean(), macro_f1]
df.loc['micro'] = [micro_precision, micro_recall, micro_f1]

#print results
print(df)