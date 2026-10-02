# ============================================================================
# 模板③（改进版）：202112 不当评论 6 标签多分类
# 相对旧模板的两个提升（实测 Macro-F1 0.582 -> 0.635）：
#   1) 每标签独立搜最优阈值（稀有标签 severe_toxic/threat/identity_hate 靠这个涨）
#   2) 加 char_wb(3,5) 字符 n-gram（抓脏话拼写变体/乱码变体，如 "sh!t" "f u c k"）
# 评分：各标签"1"的 Macro-F1，越大越好（排名制：前5%满分）
# ============================================================================
import warnings, re, sys, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.sparse import hstack

# ===== 改这里：路径 / 文本列 / ID列 / 标签列 =====
train = pd.read_csv('data/training.csv', encoding='ISO-8859-1')
test  = pd.read_csv('data/test-Ans.csv',  encoding='ISO-8859-1')   # 本地自评用带答案版；交卷时应换纯测试集
TEXT_COL = 'comment_text'
ID_COL   = 'id'
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# ====================================================

def clean(t):
    if not isinstance(t, str):
        return ''
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'&[a-zA-Z#0-9]+;', ' ', t)
    t = t.lower()
    t = re.sub(r'\W', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

tr_txt = train[TEXT_COL].apply(clean)
te_txt = test[TEXT_COL].apply(clean)
all_txt = pd.concat([tr_txt, te_txt])

# 词 n-gram + 字符 n-gram 两套特征拼一起
word = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=30000)
char = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=80000)
word.fit(all_txt); char.fit(all_txt)
X_train = hstack([word.transform(tr_txt), char.transform(tr_txt)]).tocsr()
X_test  = hstack([word.transform(te_txt), char.transform(te_txt)]).tocsr()

submission = pd.DataFrame({ID_COL: test[ID_COL]})
for col in LABELS:
    y = train[col].astype(int).values
    # 留 20% 验证集搜最优阈值（仅用于搜阈值，最终用全量重训）
    X_a, X_b, y_a, y_b = train_test_split(X_train, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear')
    clf.fit(X_a, y_a)
    pv = clf.predict_proba(X_b)[:, 1]
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(y_b, (pv >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    clf_full = LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear')
    clf_full.fit(X_train, y)
    sub = (clf_full.predict_proba(X_test)[:, 1] >= best_t).astype(int)
    submission[col] = sub
    print(f'{col:14s} 阈值={best_t:.2f}  验证F1={best_f:.4f}  正类比例={y.mean():.3f}')

submission.to_csv('考生姓名_results.csv', index=False)
print('\n提交文件已写出，前几行：')
print(submission.head())

# ---- 本地自评（若 test 带答案列，自动算 Macro-F1）----
if all(c in test.columns for c in LABELS):
    m = test[['id'] + LABELS].merge(submission, on='id', suffixes=('_t', '_p'))
    fs = [f1_score(m[c + '_t'], m[c + '_p'], zero_division=0) for c in LABELS]
    for c, f in zip(LABELS, fs):
        print(f'  {c:14s} F1={f:.4f}')
    print(f'  ** Macro-F1 = {np.mean(fs):.4f} **')
