# ============================================================================
# 12A_速通版_202609.py  —— 30 分钟保底交付（只留真正必需的代码）
# ============================================================================
# 用途：考场上先跑这份，确保「有结果能交」；之后再从 12_模板_202609结构化实战模板.py
#       里按需抄「功能段 B/C/D/E」加强，冲 F1 0.50 / 0.56 两档。
#
# 这份为什么短：去掉了行统计、两两衍生、目标编码、调参、集成。
#      它只做"能交卷 + 不掉到 0 分"必须的事：
#        读数据 → 认目标 → 补缺失/缩尾 → 编码 → LGBM 3 折 OOF → 搜门坎 → 写 CSV
#      24,000 行实测约 1~2 分钟跑完。
#
# 使用：改下面 4 行配置 → python 12A_速通版_202609.py
# 只有 3 处必须改：TRAIN_PATH / TEST_PATH / TARGET
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

# ======================= 只改这一段 =======================
TRAIN_PATH = 'train.csv'                 # 训练集
TEST_PATH = 'test.csv'                   # 测试集
TARGET = 'Revenue'                       # 目标列名（照 docx 字段表抄）
ID_COL = 'id'                            # 提交用的 ID 列名
SUB_PATH = '考生姓名_results.csv'         # 改成「你的姓名_results.csv」
SUB_COL = 'Predicted_Results'            # 预测列名（照 docx 抄）
POS_LABEL, NEG_LABEL = 'TRUE', 'FALSE'   # 正负类文字：TRUE/FALSE 或 Yes/No
SEED = 42
# =========================================================

P = lambda *a: print(*a, flush=True)


def read_table(path):
    """csv/xlsx 都能读，编码自动兜底（utf-8-sig → gb18030 → latin-1）。"""
    if str(path).lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    for enc in ['utf-8-sig', 'gb18030', 'latin-1']:
        try:
            return pd.read_csv(path, encoding=enc, na_values=['?', '.', '', ' ', 'None'])
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding='latin-1')


train, test = read_table(TRAIN_PATH), read_table(TEST_PATH)
P('train', train.shape, 'test', test.shape)
P('目标分布:\n', train[TARGET].value_counts(dropna=False).to_string())

# ---------- 1. 目标列统一成 0/1（正类 = 评分那一类）----------
s = train[TARGET]
if s.dtype == object or str(s.dtype) in ('category', 'string'):
    low = s.astype(str).str.strip().str.lower()
    if set(low.unique()) <= {'true', 'false'}:
        y = low.eq('true').astype(int)
    elif set(low.unique()) <= {'yes', 'no'}:
        y = low.eq('yes').astype(int)
    else:
        y = pd.to_numeric(s, errors='coerce')
        y = (y == sorted(y.dropna().unique())[-1]).astype(int)
else:
    y = pd.to_numeric(s, errors='coerce').astype(int)
train[TARGET] = y
P('★ 正类比例 = %.4f（题面说 202609 约 0.35）' % y.mean())

# ---------- 2. 合并 train/test，统一清洗（保证两边属性个数相同）----------
test[TARGET] = np.nan
full = pd.concat([train, test], axis=0, ignore_index=True)
is_tr = full[TARGET].notna()

# 2.1 抢救被读成文本的数值列（金额混了 $ , % 空格）
for c in full.columns:
    if c in (TARGET, ID_COL) or full[c].dtype != object:
        continue
    s2 = (full[c].astype(str).str.strip()
          .str.replace(r'[,$¥￥%\s]', '', regex=True))
    s2 = s2.replace(['', 'nan', 'None', 'NULL', '?', '.'], np.nan)
    conv = pd.to_numeric(s2, errors='coerce')
    if full[c].notna().sum() and (conv.notna() & full[c].notna()).sum() / full[c].notna().sum() >= 0.9:
        full[c] = conv

# 2.2 文本列去空格，空白当缺失
for c in full.columns:
    if full[c].dtype == object and c != ID_COL:
        full[c] = full[c].astype(str).str.strip()
        full.loc[full[c].isin(['', 'nan', 'None', 'NULL', '?', '.']), c] = np.nan

num_cols = [c for c in full.columns
            if c not in (TARGET, ID_COL) and pd.api.types.is_numeric_dtype(full[c])]
cat_cols = [c for c in full.columns
            if c not in (TARGET, ID_COL) and not pd.api.types.is_numeric_dtype(full[c])]

# 布尔列（Weekend / 是否类）统一转 0/1 —— 否则分位数缩尾时 numpy 会报错
for c in num_cols:
    if str(full[c].dtype) == 'bool' or str(full[c].dtype) == 'boolean':
        full[c] = full[c].astype(float)

# 2.3 数值列：缺失指示 + 训练集中位数填充 + 缩尾（离群值）
for c in num_cols:
    if 0.005 < full[c].isna().mean() < 0.40:
        full[c + '_isna'] = full[c].isna().astype(int)
    med = full.loc[is_tr, c].median()
    full[c] = pd.to_numeric(full[c], errors='coerce').fillna(med)
    lo, hi = full.loc[is_tr, c].quantile(0.001), full.loc[is_tr, c].quantile(0.999)
    if lo < hi:
        full[c] = full[c].clip(lo, hi)

# 2.4 类别列：缺失填 __MISSING__，再 LabelEncoder（train+test 一起 fit，码表一致）
for c in cat_cols:
    full[c] = full[c].fillna('__MISSING__').astype(str)
    full[c] = LabelEncoder().fit_transform(full[c])

feat_cols = [c for c in full.columns if c not in (TARGET, ID_COL)]
X = full.loc[is_tr, feat_cols].astype(float)
y = full.loc[is_tr, TARGET].astype(int).values
X_test = full.loc[~is_tr, feat_cols].astype(float)
test_ids = full.loc[~is_tr, ID_COL]
assert list(X.columns) == list(X_test.columns)
P('特征数 %d | X %s | X_test %s' % (len(feat_cols), X.shape, X_test.shape))

# ---------- 3. LGBM 3 折 OOF（快），不平衡用 scale_pos_weight ----------
spw = (y == 0).sum() / max((y == 1).sum(), 1)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
oof = np.zeros(len(y))
test_p = np.zeros(len(X_test))
for tr, va in skf.split(X, y):
    Xa, Xb, ya, yb = train_test_split(X.iloc[tr], y[tr], test_size=0.15,
                                      stratify=y[tr], random_state=SEED)
    m = LGBMClassifier(learning_rate=0.03, num_leaves=63, min_child_samples=20,
                       subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                       scale_pos_weight=spw, n_estimators=1500,
                       random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc',
          callbacks=[__import__('lightgbm').early_stopping(80, verbose=False)])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    test_p += m.predict_proba(X_test)[:, 1] / 3

# ---------- 4. 门坎优化（不写死 0.05~0.95，用实测概率点精确搜）----------
best_t, best_f = 0.5, -1
for t in np.unique(oof):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f:
        best_f, best_t = f, float(t)
P('默认门坎0.5 F1=%.4f | 最优门坎 %.4f F1=%.4f'
  % (f1_score(y, (oof >= 0.5).astype(int)), best_t, best_f))

# ---------- 5. 写提交（列名/取值大小写必须和 docx 一字不差）----------
pred = (test_p >= best_t).astype(int)
lab = [POS_LABEL if p == 1 else NEG_LABEL for p in pred]
out = pd.DataFrame({ID_COL: np.asarray(test_ids), SUB_COL: lab})
out.to_csv(SUB_PATH, index=False, encoding='utf-8-sig')

vals = sorted(out[SUB_COL].unique())
P('\n===== 提交自检 =====')
P('文件: %s | 列名 %s | 行数 %d' % (SUB_PATH, list(out.columns), len(out)))
P('取值: %s（必须是 [%s, %s]）' % (vals, NEG_LABEL, POS_LABEL))
P('正类占比: %.4f' % (out[SUB_COL] == POS_LABEL).mean())
assert set(vals) <= {POS_LABEL, NEG_LABEL}, '取值大小写不对！'
assert out[SUB_COL].nunique() >= 2, '全部预测同一个值 = 0 分！'
assert len(out) == len(X_test), '行数不对！'
P('✓ 格式检查通过，可以交卷。想再提分请看 12_模板_202609结构化实战模板.py 的功能段 B~E')
