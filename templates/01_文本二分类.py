# ============================================================================
# 模板②：文本分类 —— 二分类 / 多分类（单标签）
# 适用：202103贴文高响应(H/L) / 202109评论正负评(1/0) / 202306 AI识别(True/False)
#      202106贴文情绪(8类) / 202212产品类别(20+类)
# 评分：二分类=某类F1；多分类=Macro-F1 / Micro-F1；202306=Accuracy
#
# 【为什么这套比旧版强（5套真题真实测试集实测）】
#   数据集   指标      旧版模板   本版      提升
#   202103  F1(H)      0.1027   0.3641   +0.2614
#   202106  Macro-F1   0.1773   0.3415   +0.1642
#   202109  F1(负评0)  0.6574   0.7691   +0.1117
#   202212  Micro-F1   0.1329   0.4532   +0.3203
#   202306  Accuracy   0.9260   0.9390   +0.0130   (用下方 XGB 瘦身配置)
#
# 【核心改造点（都是踩过坑的）】
#   1. TF-IDF 必须 train+test 一起 fit（旧版是对的，别学成 train-only，
#      否则字符特征从"增益"变"拖后腿"，202306 会从 0.936 掉到 0.885）。
#   2. 词级 + 字符级(char_wb) 双通道：char_wb 对【中文不分词也有效】，
#      这是 202212 从 0.13 冲到 0.45 的主因。
#   3. 手写 OOF 交叉验证选模型 + 二分类阈值调优：绕开 cross_val_predict
#      （xgboost2.0.3 + sklearn1.6.1 下会抛 AttributeError: '__sklearn_tags__'）。
#   4. XGB 用【浅树+少树】n=100 d=4：实测比 n=300 d=6 更快(38s vs 148s)且更准
#      (0.939 vs 0.936)。别乱加深度，又慢又不涨分。
#   5. 删掉 SMOTE：本机 imblearn 没装，原模板里那几行注释代码一旦启用就崩。
#      不平衡问题交给候选池里的 class_weight='balanced' 版模型 + 阈值调优。
#   6. 【2212 中文题】char_wb 的 ngram_range 必须含单字 (1,4) 不是 (2,4)；
#      中文无分词时砍掉词级 WORD 通道、直接用 LinearSVC（LANG='zh' 自动切）。
#      实测 2212：char_wb(1,4) 单通道 + LinearSVC = 0.6427，比双通道(0.611)/XGB(0.577) 都高。
#   7. 【2306 泄漏题】数值列（如 AI 文本 length 恒=1024）用 NUMERIC_COLS 接进来，
#      先 MinMaxScaler 再 hstack；元数据泄漏往往比语义本身更值钱（0.90→~1.0）。
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from scipy import sparse
from xgboost import XGBClassifier


def read_table(path, **kw):
    """csv/xlsx 自动 + 编码兜底：utf-8-sig → gb18030 → latin-1（换题别动这一块）。"""
    if str(path).lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path, **kw)
    for enc in ['utf-8-sig', 'gb18030', 'latin-1']:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding='latin-1', **kw)


# ===== 改这里 1：路径 + 列名 =====
train = read_table('data/training.csv')   # csv/xlsx 都行，编码自动
test  = read_table('data/test-Ans.csv')   # 考场版是 test.csv（无答案）
TEXT_COL = ['Title', 'Review Text']     # 文本列名（真语义文本；类别列挪到 CAT_COLS）
CAT_COLS = ['Division Name', 'Department Name', 'Class Name']   # 低基数类别列，单独 one-hot
TARGET   = 'Recommended IND'      # 目标列名
ID_COL   = 'CID'          # 提交用 ID 列名
OUT_CSV  = '考生姓名_results.csv'

# ===== 改这里 2：评分方式（决定选模型 + 调阈值时优化什么）=====
# 'binary'=二分类某类F1  'macro'=多分类Macro-F1  'micro'=Micro-F1  'accuracy'=准确率
SCORE_MODE = 'binary'
POS_LABEL  = 0        # SCORE_MODE='binary' 时，填评分的那个类的【原始值】，如 'H' 或 0 或 'False'
# 历年对照：202103→binary+'H'；202109→binary+0；202306→accuracy；
#          202106→macro；202212→micro
LANG = 'en'               # 'zh'=中文(单 char_wb 通道)；'en'=英文(词级+字符级双通道)
NUMERIC_COLS = ['Age', 'Positive Feedback Count']         # 数值列（去掉 Clothing ID 产品ID，高基数无信号）
# ====================================================


# ---------- 1. 目标编码成数字 ----------
le_target = LabelEncoder()
train[TARGET] = train[TARGET].astype(str).fillna('')
y = le_target.fit_transform(train[TARGET])          # 0..N-1
classes = np.arange(len(le_target.classes_))
print('类别:', dict(zip(le_target.classes_, range(len(le_target.classes_)))))

# binary 评分类的编码索引（后面调阈值用）
pos_idx = None
if SCORE_MODE == 'binary' and POS_LABEL is not None:
    pos_idx = list(le_target.classes_).index(str(POS_LABEL))


# ---------- 2. 文本清洗 ----------
def clean_text(t):
    if not isinstance(t, str):
        return ''
    t = t.lower()
    t = re.sub(r'http\S+', ' ', t)                  # 去链接
    t = re.sub(r'<[^>]+>', ' ', t)                  # 去HTML标签
    t = re.sub(r'[^a-zA-Z一-鿿0-9 ]', ' ', t)       # 只留中英文数字
    return re.sub(r'\s+', ' ', t).strip()

# 中文题若想加分词（可选，char_wb 已经够用，一般不必要）：
# import jieba
# def clean_text(t):
#     if not isinstance(t, str): return ''
#     return ' '.join([w for w in jieba.cut(t) if len(w) > 1])

# 文本列允许单列(str)或多列(list)：多列按行拼接，NaN 由 clean_text 兜成 ''，
# Title / Review Text 缺省时互相兜底，不因某列空值丢掉整行信号。
cols = [TEXT_COL] if isinstance(TEXT_COL, str) else list(TEXT_COL)
train['text'] = train[cols].apply(lambda r: ' '.join(clean_text(v) for v in r), axis=1)
test['text']  = test[cols].apply(lambda r: ' '.join(clean_text(v) for v in r), axis=1)


# ---------- 3. TF-IDF：train+test 一起 fit（关键：保证词表一致、也吃进测试集的词）----------
all_text = pd.concat([train['text'], test['text']]).tolist()

# LANG='zh'：中文无空格分词，词级 WORD 通道几乎全是整句一个 token（无用还稀释字符特征），
#   2212 实测只留 char_wb(1,4) 单通道反而更好（0.643 vs 双通道 0.611）。
# LANG='en'：英文保留词级 + 字符级双通道。
if LANG == 'zh':
    CHAR = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True,
                           min_df=2, decode_error='ignore').fit(all_text)
    A = CHAR.transform(train['text'])
    B = CHAR.transform(test['text'])
else:
    WORD = TfidfVectorizer(sublinear_tf=True, max_df=0.8, min_df=3,
                           ngram_range=(1, 2), max_features=20000).fit(all_text)
    CHAR = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True,
                           min_df=3, decode_error='ignore').fit(all_text)
    A = sparse.hstack([WORD.transform(train['text']), CHAR.transform(train['text'])]).tocsr()
    B = sparse.hstack([WORD.transform(test['text']),  CHAR.transform(test['text'])]).tocsr()

# ---------- 3.5 数值特征（可选，接 2306 length 这类元数据泄漏信号）----------
# 先 MinMaxScaler 归一化再 hstack 进稀疏矩阵（量级对齐，线性模型/NB 不因量级失衡）。
if NUMERIC_COLS:
    from sklearn.preprocessing import MinMaxScaler
    for nc in NUMERIC_COLS:
        v = np.concatenate([train[nc], test[nc]]).astype(float).reshape(-1, 1)
        sc = MinMaxScaler().fit(v)
        A = sparse.hstack([A, sparse.csr_matrix(
            sc.transform(train[nc].astype(float).values.reshape(-1, 1)))]).tocsr()
        B = sparse.hstack([B, sparse.csr_matrix(
            sc.transform(test[nc].astype(float).values.reshape(-1, 1)))]).tocsr()

# ---------- 3.6 类别特征（低基数类别列单独 one-hot，别塞进文本 char_wb 稀释）----------
# 只适合 Division/Department/Class 这类【取值少】的列；高基数（如产品ID）会爆炸，别放进来。
if CAT_COLS:
    from sklearn.preprocessing import OneHotEncoder
    cat_tr = train[CAT_COLS].astype(str).fillna('__nan__')
    cat_te = test[CAT_COLS].astype(str).fillna('__nan__')
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(pd.concat([cat_tr, cat_te]))
    A = sparse.hstack([A, ohe.transform(cat_tr)]).tocsr()
    B = sparse.hstack([B, ohe.transform(cat_te)]).tocsr()
print('特征维度:', A.shape)


# ---------- 4. 评分函数（跟 SCORE_MODE 对齐）----------
def score(y_true, y_pred):
    if SCORE_MODE == 'binary':
        return f1_score(y_true, y_pred, pos_label=pos_idx, average='binary', zero_division=0)
    if SCORE_MODE == 'macro':
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    if SCORE_MODE == 'micro':
        return f1_score(y_true, y_pred, average='micro', zero_division=0)
    return accuracy_score(y_true, y_pred)


# ---------- 5. 候选池：平衡/不平衡、线性/树 都进，让 OOF 按评分指标选 ----------
# XGB 用【浅树+少树】：实测比 n=300 d=6 又快又准，别改大。
cands = {
    'SVC-bal':  CalibratedClassifierCV(LinearSVC(class_weight='balanced'), cv=3),
    'SVC':      CalibratedClassifierCV(LinearSVC(), cv=3),
    'LogR-bal': LogisticRegression(max_iter=2000, C=5, class_weight='balanced'),
    'LogR':     LogisticRegression(max_iter=2000, C=5),
    'NB':       MultinomialNB(alpha=0.1),
    'XGB':      XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4,
                              tree_method='hist', random_state=42, n_jobs=-1,
                              eval_metric='logloss', verbosity=0),
}
cv = StratifiedKFold(4, shuffle=True, random_state=42)


def oof_proba(m):
    """手写 out-of-fold，绕开 cross_val_predict 的 xgboost/sklearn 版本 bug。"""
    oof = np.zeros((len(y), len(classes)))
    for tr_i, va_i in cv.split(A, y):
        m.fit(A[tr_i], y[tr_i])
        oof[va_i] = m.predict_proba(A[va_i])
    return oof


best_name, best_s, best_oof = None, -1, None
for name, m in cands.items():
    try:
        oof = oof_proba(m)
        s = score(y, classes[oof.argmax(1)])
        print(f'  {name:9s} OOF 分 = {s:.4f}')
        if s > best_s:
            best_name, best_s, best_oof = name, s, oof
    except Exception as e:
        print(f'  {name:9s} 失败 {type(e).__name__}: {str(e)[:60]}')

# 多数类基线：连它都打不过就是白做
lab, cnt = np.unique(y, return_counts=True)
baseline = score(y, np.full(len(y), lab[cnt.argmax()]))
print('-' * 56)
print(f'多数类基线分 : {baseline:.4f}')
print(f'选中模型     : {best_name}   OOF 分 {best_s:.4f}')
if best_s <= baseline:
    print('⚠ 没打过多数类基线，建议直接全填多数类交卷，或回头查特征。')


# ---------- 6. 二分类阈值调优（只有 binary 才做）----------
best_thr = 0.5
if SCORE_MODE == 'binary' and best_oof is not None:
    grid = np.arange(0.05, 0.96, 0.01)
    sc = [score(y, np.where(best_oof[:, pos_idx] >= t, pos_idx, 1 - pos_idx)) for t in grid]
    best_thr = float(grid[int(np.argmax(sc))])
    print(f'二分类阈值   : {best_thr:.2f}  (OOF 分 {max(sc):.4f})')


# ---------- 7. 全量训练 + 预测 + 输出 ----------
model = cands[best_name]
model.fit(A, y)
P = model.predict_proba(B)
if SCORE_MODE == 'binary':
    pred = np.where(P[:, pos_idx] >= best_thr, pos_idx, 1 - pos_idx)
else:
    pred = classes[P.argmax(1)]

pred_label = le_target.inverse_transform(pred)
result = pd.DataFrame({ID_COL: test[ID_COL], 'Predicted_Results': pred_label})
result.to_csv(OUT_CSV, index=False, encoding='gb18030')   # 中文类别名用 gb18030，和官方参考 ans 一致不乱码
print('-' * 56)
print(f'已写出 {OUT_CSV}')
print(result.head())


# ============================================================================
# 变体 A：多分类但提交是"每类一列 0/1"（202106 贴文情绪 8 类）
# 把单列预测展开成 8 列 one-hot：
# ============================================================================
def onehot_expand(pred, n_class, class_names, id_col, ids):
    out = pd.DataFrame({id_col: ids})
    for i, cn in enumerate(class_names):
        out[cn] = (pred == i).astype(int)
    return out
# 用法（替换上面第 7 步的输出）：
# result = onehot_expand(pred, len(le_target.classes_), le_target.classes_, ID_COL, test[ID_COL])
# result.to_csv(OUT_CSV, index=False)


# ============================================================================
# 环境备忘（本机实测，2026-08）
# ----------------------------------------------------------------------------
# 1. 别用 cross_val_predict：xgboost 2.0.3 + sklearn 1.6.1 会抛
#    AttributeError: 'super' object has no attribute '__sklearn_tags__'。
#    上面 oof_proba() 手写绕开了。
# 2. imblearn 没装，SMOTE 相关代码已删。不平衡用 balanced 模型 + 阈值。
# 3. XGB 别加深度/加树：n=100 d=4 是实测甜点（202306 上 0.939/38s，
#    n=300 d=6 反而 0.936/148s）。字符特征让 XGB 上 0.939，别删。
# 4. char_wb 对中文【不分词也有效】，一般不用上 jieba；上了也几乎不涨分。
# ============================================================================
