# ============================================================================
# 模板③：多标签分类（一个样本可同时属于多个类别）
# 适用：202112 不当评论(6标签 toxic/severe_toxic/obscene/threat/insult/identity_hate)
#      202503 网站提问多标签(SQLite，Tag 逗号分隔)
# 评分：202112 = 各标签"1"的 Macro-F1；202503 = Micro-F1
# 两种做法：
#   A. 每个标签单独训一个二分类器（最简单，202112 官方解法）
#   B. MultiLabelBinarizer + OneVsRest（每个标签一个二分类器，202503 官方解法思路）
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ===== 改这里 1：路径 + 文本列 + 标签列名 =====
train = pd.read_csv('Train_data.csv', encoding='ISO-8859-1')   # 编码错误就试 utf-8 / gb18030
test  = pd.read_csv('Test_data_Ans.csv',  encoding='ISO-8859-1')
TEXT_COL = 'comment_text'
ID_COL   = 'id'
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# ====================================================

# ---------- 1. 文本清洗（英文） ----------
def clean_text(t):
    if not isinstance(t, str):
        return ''
    t = t.lower()
    t = re.sub(r'\W', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t
train['text'] = train[TEXT_COL].apply(clean_text)
test['text']  = test[TEXT_COL].apply(clean_text)

# ---------- 2. TF-IDF ----------
all_text = pd.concat([train['text'], test['text']]).tolist()
tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=30000)
tfidf.fit(all_text)
X_train_all = tfidf.transform(train['text'])
X_test = tfidf.transform(test['text'])

# ============================================================================
# 做法 A：每个标签单独二分类（推荐，稳）
# ⚠️ 本机 imblearn 没装，SMOTE 会 import 直接崩掉，已换成 class_weight='balanced'
#    （同模板②环境备忘）。tox/insult 等标签正类占比极低，balanced 权重足够对冲。
# ============================================================================
submission = pd.DataFrame({ID_COL: test[ID_COL]})
for col in LABELS:
    y_col = train[col].astype(int)
    # 留 20% 快速看一眼 F1，再用全量数据重训最终模型（比只用 80% 更稳）
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_all, y_col, test_size=0.2, random_state=42, stratify=y_col)
    clf = LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear')
    clf.fit(X_tr, y_tr)
    print(f'{col}: 验证集F1={f1_score(y_val, clf.predict(X_val), zero_division=0):.4f}'
          f'  正类比例={y_col.mean():.3f}')
    clf_full = LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear')
    clf_full.fit(X_train_all, y_col)
    submission[col] = clf_full.predict(X_test)

submission.to_csv('考生姓名_results.csv', index=False)   # 多列 0/1 直接交
print(submission.head())

# ============================================================================
# 做法 B：MultiLabelBinarizer + 每个标签一个二分类器（适合标签多、输出"逗号分隔"的题，如 202503）
# 注意：原 xgb.train(binary:logistic) 塞 2D 标签会报错，这里改成 sklearn OneVsRest 包装，稳且不依赖 xgboost。
# ============================================================================
def multilabel_ovr(train_df, test_df, text_col, id_col, label_col, split_char=','):
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multiclass import OneVsRestClassifier

    # 把标签列转成 list-of-tags：202503 是 "a,b" 逗号分隔
    tags = train_df[label_col].fillna('').astype(str).str.split(split_char)
    tags = [[t.strip() for t in ts if t.strip()] for ts in tags]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(tags)                      # 2D 0/1

    # 文本特征：train+test 一起 fit（同模板②，保证词表一致）
    tf = TfidfVectorizer(sublinear_tf=True, max_features=20000, ngram_range=(1, 2))
    tf.fit(pd.concat([train_df[text_col].fillna(''), test_df[text_col].fillna('')]))
    X_tr = tf.transform(train_df[text_col].fillna(''))
    X_te = tf.transform(test_df[text_col].fillna(''))

    # 每个标签一个 balanced 逻辑回归（无需 SMOTE）
    clf = OneVsRestClassifier(
        LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear'))
    clf.fit(X_tr, y)

    proba = clf.predict_proba(X_te)                  # (样本, 标签) 每列是 P(该标签=1)
    p = (proba >= 0.5).astype(int)
    # 无标签样本兜底：取概率最大的那个标签
    for i in range(p.shape[0]):
        if p[i].sum() == 0:
            p[i, np.argmax(proba[i])] = 1

    labels = mlb.inverse_transform(p)
    out = pd.DataFrame({id_col: test_df[id_col],
                        'Tag': [','.join(ls) for ls in labels]})
    return out

# 用法（202503 示例）：先把 SQLite 读出来合并成 training/test 两个表，再调用：
# out = multilabel_ovr(training, test, 'text', 'Id', 'Tag', split_char=',')
# out.to_csv('考生姓名_results.csv', index=False)
