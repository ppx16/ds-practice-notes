# -*- coding: utf-8 -*-
# ============================================================================
# 模板③（通用版）：多标签分类 —— 一个样本可同时属于多个标签
#
# 适用（历年真题）：
#   202112 不当评论 6 标签（toxic/severe_toxic/... 多列 0/1）  → 各标签 Macro-F1
#   202503 网站提问多标签（Tag 逗号分隔单列 "a,b,c"）           → Micro-F1
#   ……以及任何「一个样本 → 多个标签」的题。
#
# 用法：只改下面 CONFIG 里那几行，其余照抄不用动。
#   TEXT_COL / ID_COL / LABELS 填 'auto' 会自动猜，猜完会 print 出来给你核对；
#   核对不对就把 'auto' 换成真实列名（写死最稳，考试优先写死）。
#
#   LABELS 三种填法：
#     'auto'                        → 自动找所有 0/1 列（多列 one-hot，如 202112）
#     ['toxic','severe_toxic',...]  → 显式多列 one-hot
#     'Tag'                         → 单列逗号分隔标签（如 202503 的 "a,b,c"）
#
#   SCORE：'macro'（202112 各标签 Macro-F1）/ 'micro'（202503 Micro-F1）
#
# 关键机制（都是实测踩出来的，别删）：
#   1) 每标签独立搜最优阈值：稀有标签靠高阈值换 precision（202112 0.582 → 0.635）
#   2) class_weight='balanced'：对冲稀有标签正类极少（本机无 imblearn，别装 SMOTE）
#   3) 中英文自动分流：中文 char_wb(1,4)（必须含 1，单字是主增益）；
#      英文 word(1,2) + char_wb(3,5)（char 抓脏话拼写变体）
# ============================================================================
import warnings, re, sys
warnings.filterwarnings('ignore')
try:
    # 修 Windows GBK 控制台打印中文报错；line_buffering=True 让进度实时显示（别憋到最后）
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack

# ==================== 只改这里（其余照抄） ====================
CONFIG = dict(
    TRAIN_PATH = 'data/training.csv',       # 训练数据路径（.csv / .xlsx 都行）
    TEST_PATH  = 'data/test-Ans.csv',       # 考场版换成【无答案列】的纯测试集
    TEXT_COL   = 'auto',                    # 文本列名；'auto'=自动猜
    ID_COL     = 'auto',                    # ID 列名；'auto'=自动猜
    LABELS     = 'auto',                    # 见文件头：'auto' / 列表 / 单列名
    SCORE      = 'macro',                   # 'macro'（202112）/ 'micro'（202503）
    OUT_CSV    = '考生姓名_results.csv',
)
# ============================================================


# ---------------- 读数据：编码自动试 ----------------
def read_table(path):
    if str(path).lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    for enc in ['utf-8-sig', 'gb18030', 'latin-1']:   # latin-1 兜底永不报错
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding='latin-1')


# ---------------- 列名自动猜（猜错就在 CONFIG 写死） ----------------
def guess_text_col(df):
    for g in ['comment_text', 'text', 'content', 'description', 'Product_Description',
              'Content', 'Text', '正文', '文本', '内容', '微博内容', '问题', '问题描述']:
        if g in df.columns:
            return g
    obj = df.select_dtypes(include='object').columns
    if len(obj) == 0:
        return None
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())  # 字符串平均最长的列


def guess_id_col(df):
    for g in ['id', 'ID', 'Id', 'Id', 'Product_ID', 'user_id', 'video_id']:
        if g in df.columns:
            return g
    for c in df.columns:
        if df[c].nunique() == len(df):
            return c
    return df.columns[0]


def guess_label_cols(df, id_col, text_col):
    cols = []
    for c in df.columns:
        if c in (id_col, text_col):
            continue
        vals = set(df[c].dropna().unique())
        if len(vals) == 2 and vals <= {0, 1, '0', '1'}:
            cols.append(c)
    return cols


# ---------------- 文本清洗：中英文分流 ----------------
def _is_chinese(series):
    s = series.dropna().astype(str).head(500)
    return s.str.contains(r'[一-鿿]').mean() > 0.3


def clean_en(t):
    if not isinstance(t, str):
        return ''
    t = re.sub(r'<[^>]+>', ' ', t)             # 去 HTML 标签
    t = re.sub(r'&[a-zA-Z#0-9]+;', ' ', t)     # 去 &nbsp; &lt; 等实体
    t = t.lower()
    t = re.sub(r'\W', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def clean_zh(t):
    if not isinstance(t, str):
        return ''
    return re.sub(r'[^一-鿿a-zA-Z0-9]', ' ', t).strip()  # 只留中文+英文+数字


# ---------------- 特征：train+test 一起 fit，保证词表一致 ----------------
def build_features(all_text, is_zh):
    if is_zh:
        tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4),
                             sublinear_tf=True, min_df=2, decode_error='ignore')
        tf.fit(all_text)
        return tf.transform(all_text), tf.get_feature_names_out().shape[0]
    word = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=30000)
    char = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(3, 5),
                           min_df=3, max_features=80000)
    word.fit(all_text)
    X = word.transform(all_text)
    char.fit(all_text)
    X = hstack([X, char.transform(all_text)]).tocsr()
    return X, X.shape[1]


# ---------------- 核心：每标签搜阈值 + balanced 逻辑回归 ----------------
def run_multilabel(X_train, X_test, y, label_names, score):
    n_lab = y.shape[1]
    preds = np.zeros((X_test.shape[0], n_lab), dtype=int)
    for j in range(n_lab):
        yj = y[:, j].astype(int)
        stratify = yj if (yj.sum() >= 10) else None          # 太稀就放弃分层
        X_a, X_b, ya, yb = train_test_split(X_train, yj, test_size=0.2,
                                            random_state=42, stratify=stratify)
        clf = LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear')
        clf.fit(X_a, ya)
        if yj.sum() < 10:                          # 正例太少，搜阈值会过拟合，固定 0.5
            best_t, best_f = 0.5, -1.0
        else:
            pv = clf.predict_proba(X_b)[:, 1]
            best_t, best_f = 0.5, -1.0
            for t in np.arange(0.05, 0.96, 0.01):
                f = f1_score(yb, (pv >= t).astype(int), zero_division=0)
                if f > best_f:
                    best_f, best_t = f, t
        clf_full = LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear')
        clf_full.fit(X_train, yj)
        preds[:, j] = (clf_full.predict_proba(X_test)[:, 1] >= best_t).astype(int)
        print(f'  {str(label_names[j]):14s} 阈值={best_t:.2f}  验证F1={best_f:.4f}  正类比例={yj.mean():.3f}')
    return preds


# ---------------- 主流程 ----------------
def main():
    cfg = CONFIG
    train = read_table(cfg['TRAIN_PATH'])
    test = read_table(cfg['TEST_PATH'])
    print(f'训练 {train.shape}   测试 {test.shape}')

    text_col = cfg['TEXT_COL'] if cfg['TEXT_COL'] != 'auto' else guess_text_col(train)
    id_col = cfg['ID_COL'] if cfg['ID_COL'] != 'auto' else guess_id_col(train)

    # ---- 判断标签形态：多列 one-hot 还是逗号分隔单列 ----
    is_comma = isinstance(cfg['LABELS'], str) and cfg['LABELS'] != 'auto' and cfg['LABELS'] in train.columns
    if is_comma:
        tags = train[cfg['LABELS']].fillna('').astype(str).str.split(',')
        tags = [[t.strip() for t in ts if t.strip()] for ts in tags]
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(tags)
        label_names = list(mlb.classes_)
        label_cols = None
    else:
        label_cols = (guess_label_cols(train, id_col, text_col)
                      if cfg['LABELS'] == 'auto' else list(cfg['LABELS']))
        y = train[label_cols].astype(int).values
        label_names = label_cols
        mlb = None

    print(f'文本列={text_col}  ID列={id_col}  标签数={len(label_names)}  '
          f'标签形态={"逗号分隔" if is_comma else "多列one-hot"}')
    print('标签:', label_names)

    # ---- 清洗 + 特征 ----
    is_zh = _is_chinese(train[text_col])
    clean = clean_zh if is_zh else clean_en
    print('语言:', '中文(走 char_wb(1,4))' if is_zh else '英文(走 word+char)')
    tr_txt = train[text_col].fillna('').apply(clean)
    te_txt = test[text_col].fillna('').apply(clean)
    n_tr = len(tr_txt)
    all_text = pd.concat([tr_txt, te_txt])
    X_all, n_feat = build_features(all_text, is_zh)
    X_train, X_test = X_all[:n_tr], X_all[n_tr:]
    print(f'特征维度={n_feat}')

    # ---- 训练 + 预测 ----
    preds = run_multilabel(X_train, X_test, y, label_names, cfg['SCORE'])

    # ---- 写出提交 ----
    if is_comma:
        # 无标签样本兜底：全 0 就取概率最大的那个标签（202503 这样才稳）
        for i in range(preds.shape[0]):
            if preds[i].sum() == 0:
                preds[i, preds[i].argmax()] = 1
        out = pd.DataFrame({id_col: test[id_col],
                            'Tag': [','.join(mlb.classes_[preds[i].astype(bool)])
                                    for i in range(preds.shape[0])]})
    else:
        out = pd.DataFrame({id_col: test[id_col]})
        for k, c in enumerate(label_names):
            out[c] = preds[:, k]
    out.to_csv(cfg['OUT_CSV'], index=False)
    print(f'\n已写出 {cfg["OUT_CSV"]}  ({len(out)} 行)，前几行：')
    print(out.head())

    # ---- 本地自评（测试集带答案列时自动算分） ----
    avg = 'micro' if cfg['SCORE'] == 'micro' else 'macro'
    if is_comma and cfg['LABELS'] in test.columns:
        tt = test[cfg['LABELS']].fillna('').astype(str).str.split(',')
        tt = [[s.strip() for s in ts if s.strip()] for ts in tt]
        y_true = mlb.transform(tt)
        print(f'  ** {cfg["SCORE"].upper()}-F1 = '
              f'{f1_score(y_true, preds, average=avg, zero_division=0):.4f} **')
    elif not is_comma and all(c in test.columns for c in label_cols):
        fs = [f1_score(test[c].astype(int), preds[:, k], zero_division=0)
              for k, c in enumerate(label_names)]
        for c, f in zip(label_names, fs):
            print(f'  {c:14s} F1={f:.4f}')
        print(f'  ** {cfg["SCORE"].upper()}-F1 = '
              f'{f1_score(test[label_cols].astype(int), preds, average=avg, zero_division=0):.4f} **')


if __name__ == '__main__':
    main()
