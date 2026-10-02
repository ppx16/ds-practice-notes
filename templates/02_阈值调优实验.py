# 实验：量化「阈值调优」和「char n-gram 特征」对 Macro-F1 的提升
# 结论先行：基线(硬0.5)≈0.582；调阈值后稀有标签(severe_toxic/threat/identity_hate)明显上涨
import warnings, re, sys, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.sparse import hstack

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
train = pd.read_csv('data/training.csv', encoding='ISO-8859-1')
test  = pd.read_csv('data/test-Ans.csv',  encoding='ISO-8859-1')
ans   = test[['id'] + LABELS]

def clean(t):
    if not isinstance(t, str): return ''
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'&[a-zA-Z#0-9]+;', ' ', t)
    t = t.lower()
    t = re.sub(r'\W', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

tr_txt = train['comment_text'].apply(clean)
te_txt = test['comment_text'].apply(clean)
all_txt = pd.concat([tr_txt, te_txt])

def macro_f1(pred):
    m = ans.merge(pred[['id']+LABELS], on='id', suffixes=('_t','_p'))
    return np.mean([f1_score(m[c+'_t'], m[c+'_p'], zero_division=0) for c in LABELS])

def run(features, name, tune=True):
    Xtr, Xte = features
    sub = pd.DataFrame({'id': test['id']})
    for col in LABELS:
        y = train[col].astype(int).values
        X_a, X_b, y_a, y_b = train_test_split(Xtr, y, test_size=0.2, random_state=42, stratify=y)
        clf = LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear')
        clf.fit(X_a, y_a)
        pv = clf.predict_proba(X_b)[:, 1]
        thr = 0.5
        if tune:
            best_t, best_f = 0.5, -1.0
            for t in np.arange(0.05, 0.96, 0.01):
                f = f1_score(y_b, (pv >= t).astype(int), zero_division=0)
                if f > best_f: best_f, best_t = f, t
            thr = best_t
        clf_full = LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear')
        clf_full.fit(Xtr, y)
        sub[col] = (clf_full.predict_proba(Xte)[:, 1] >= thr).astype(int)
    print(f'[{name}] Macro-F1 = {macro_f1(sub):.4f}')

# 特征 1：word tfidf (1,2) —— 和现模板一致
word = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=3, max_features=30000)
word.fit(all_txt)
Xtr_w = word.transform(tr_txt); Xte_w = word.transform(te_txt)

# 特征 2：word + char_wb(3,5) —— 抓拼写变体/脏话变体
char = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(3,5), min_df=3, max_features=80000)
char.fit(all_txt)
Xtr_c = hstack([Xtr_w, char.transform(tr_txt)]).tocsr()
Xte_c = hstack([Xte_w, char.transform(te_txt)]).tocsr()

run((Xtr_w, Xte_w), 'word(1,2) 硬0.5', tune=False)
run((Xtr_w, Xte_w), 'word(1,2) 调阈值', tune=True)
run((Xtr_c, Xte_c), 'word+char(3-5) 调阈值', tune=True)
