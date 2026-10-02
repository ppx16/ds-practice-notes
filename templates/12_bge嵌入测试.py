# -*- coding: utf-8 -*-
# 对比实验：TF-IDF 基线 vs 加 bge-m3 语义相似度特征（替代官方解的 BERT）
import warnings; warnings.filterwarnings('ignore')
import ssl
_orig = ssl.SSLContext.load_default_certs
def _patched(self, *a, **k):
    try: _orig(self, *a, **k)
    except Exception: pass
ssl.SSLContext.load_default_certs = _patched

import re
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix

train = pd.read_excel('data/training.xlsx').merge(
        pd.read_excel('data/all_articles.xlsx'), on='article_id', how='left')
test  = pd.read_excel('data/test.xlsx').merge(
        pd.read_excel('data/all_articles.xlsx'), on='article_id', how='left')

# ---------- 与模板相同的特征 ----------
def text_stats(text):
    t = '' if pd.isna(text) else str(text)
    words = t.split()
    sents = [s for s in re.split(r'[.!?]+', t) if s.strip()]
    n_char, n_word = len(t), len(words)
    n_sent = len(sents) if sents else 1
    n_uniq = len(set(w.lower() for w in words))
    return dict(num_chars=n_char, num_words=n_word, num_sentences=n_sent,
                avg_word_length=n_char/n_word if n_word else 0,
                avg_sentence_length=n_word/n_sent if n_sent else 0,
                unique_word_ratio=n_uniq/n_word if n_word else 0)

def grammar_error_count(text):
    t = '' if pd.isna(text) else str(text)
    err = 0
    words = t.lower().split()
    for i in range(len(words)-1):
        if words[i] == words[i+1]: err += 1
    for s in re.split(r'(?<=[.!?])\s+', t.strip()):
        s = s.strip()
        if s and not s[0].isupper(): err += 1
        if len(s.split()) > 40: err += 1
    for a,b in [('(',')'),('[',']'),('{','}')]:
        if t.count(a) != t.count(b): err += 1
    return err

all_texts = pd.concat([train['summary_text'], train['article_text'],
                       train['summary_question']], axis=0)
tfidf_rel = TfidfVectorizer(stop_words='english', max_features=3000)
tfidf_rel.fit(all_texts.fillna(''))

def cos_row(X, Y):
    X = normalize(X, norm='l2'); Y = normalize(Y, norm='l2')
    return np.asarray(X.multiply(Y).sum(axis=1)).ravel()

def rel_features(df):
    s = tfidf_rel.transform(df['summary_text'].fillna(''))
    a = tfidf_rel.transform(df['article_text'].fillna(''))
    q = tfidf_rel.transform(df['summary_question'].fillna(''))
    return cos_row(s,a), cos_row(s,q)

tfidf_bow = TfidfVectorizer(stop_words='english', max_features=4000,
                            ngram_range=(1,2), min_df=2)
tfidf_bow.fit(train['summary_text'].fillna(''))

art_cats = sorted(pd.concat([train['article_id'], test['article_id']]).unique())

def build_features(df):
    stats = df['summary_text'].apply(text_stats).apply(pd.Series)
    stats['gram_err'] = df['summary_text'].apply(grammar_error_count)
    sim_a, sim_q = rel_features(df)
    stats['sim_article'] = sim_a
    stats['sim_question'] = sim_q
    art = pd.get_dummies(pd.Categorical(df['article_id'], categories=art_cats),
                         prefix='art').astype(int)
    return pd.concat([stats.reset_index(drop=True), art.reset_index(drop=True)], axis=1)

feat_train = build_features(train)
feat_test  = build_features(test)
X_bow_train = tfidf_bow.transform(train['summary_text'].fillna(''))
X_bow_test  = tfidf_bow.transform(test['summary_text'].fillna(''))
X_base      = hstack([csr_matrix(feat_train.values), X_bow_train]).tocsr()
X_base_test = hstack([csr_matrix(feat_test.values),  X_bow_test]).tocsr()
y = train['score'].astype(float).values

# ---------- bge-m3 语义相似度特征 ----------
print('加载 bge-m3 ...')
model = SentenceTransformer(r'D:/学习资料/练习/models/bge-m3', device='cuda')

def emb(texts):
    return model.encode(list(texts), normalize_embeddings=True,
                        batch_size=64, show_progress_bar=False)

# 4 篇原文 / 4 个问题 的 embedding（按 article_id 缓存）
art_ids = sorted(train['article_id'].unique())
art_emb, q_emb = {}, {}
for aid in art_ids:
    row = train[train['article_id'] == aid].iloc[0]
    art_emb[aid] = emb([row['article_text']])[0]
    q_emb[aid]   = emb([row['summary_question']])[0]

def bge_features(df):
    e = emb(df['summary_text'].fillna(''))
    sim_a = np.array([float(e[i] @ art_emb[aid]) for i, aid in enumerate(df['article_id'])])
    sim_q = np.array([float(e[i] @ q_emb[aid])   for i, aid in enumerate(df['article_id'])])
    return sim_a, sim_q

print('编码 bge-m3 ...')
bge_tr  = bge_features(train)
bge_te  = bge_features(test)
B_tr  = np.vstack(bge_tr).T
B_te  = np.vstack(bge_te).T

X_full      = hstack([X_base, csr_matrix(B_tr)]).tocsr()
X_full_test = hstack([X_base_test, csr_matrix(B_te)]).tocsr()

# ---------- 对比两个模型的 5 折 OOF ----------
def run_cv(X, name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = LGBMRegressor(objective='regression_l1', metric='rmse',
                          n_estimators=1500, learning_rate=0.05, num_leaves=31,
                          colsample_bytree=0.8, subsample=0.8,
                          reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va])
    return mean_squared_error(y, oof)

mse_base = run_cv(X_base, 'TF-IDF 基线')
mse_bge  = run_cv(X_full, 'TF-IDF + bge-m3')
print('TF-IDF 基线        OOF MSE = %.4f' % mse_base)
print('TF-IDF + bge-m3    OOF MSE = %.4f' % mse_bge)
