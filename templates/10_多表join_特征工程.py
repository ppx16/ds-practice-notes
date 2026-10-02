# ============================================================================
# 202412 欺诈检测 —— 多表 join + 特征工程(细化版) + 二分类
# 本版在上一版基础上追加三件事：
#   (1) 交叉特征：SuspiciousFlag × 金额/大额/余额比/深夜
#   (2) XGB/LGBM 加早停(early stopping) + 更细调参
#   (3) 特征重要性排序，看哪些特征真在起作用
# 评分：正类(欺诈=1) 的 F1；输出 0/1；运行位置：练习2412 目录
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping as lgbm_es

pd.set_option('display.max_columns', None)

# ===== 改这里 1：数据路径 + 目标列名 + ID 列名 =====
transaction_records = pd.read_csv("data/Transaction Data/transaction_records.csv")
transaction_metadata = pd.read_csv("data/Transaction Data/transaction_metadata.csv")
customer_data = pd.read_csv("data/Customer Profiles/customer_data.csv")
account_activity = pd.read_csv("data/Customer Profiles/account_activity.csv")
merchant_data = pd.read_csv("data/Merchant Information/merchant_data.csv")
transaction_category_labels = pd.read_csv("data/Merchant Information/transaction_category_labels.csv")
suspicious_activity = pd.read_csv("data/Fraudulent Patterns/suspicious_activity.csv")

TARGET = 'FraudIndicator'
ID_COL = 'TransactionID'

# ---------- 多表 join ----------
merged_data = transaction_records.merge(transaction_metadata, on="TransactionID", how="left")
merged_data = merged_data.merge(customer_data, on="CustomerID", how="left")
merged_data = merged_data.merge(account_activity, on="CustomerID", how="left")
merged_data = merged_data.merge(merchant_data, on="MerchantID", how="left")
merged_data = merged_data.merge(transaction_category_labels, on="TransactionID", how="left")
merged_data = merged_data.merge(suspicious_activity, on="CustomerID", how="left")

train = pd.read_csv("data/train_data.csv")
test = pd.read_csv("data/test_data.csv")
train[TARGET] = train[TARGET].astype(int)
train = train.merge(merged_data, on=ID_COL, how="left")
test = test.merge(merged_data, on=ID_COL, how="left")
# ====================================================

test[TARGET] = np.nan
full = pd.concat([train, test], axis=0, ignore_index=True)

# ================= 特征工程 =================
full['Timestamp'] = pd.to_datetime(full['Timestamp'])
# full['LastLogin'] = pd.to_datetime(full['LastLogin'])
# full = full.sort_values(['CustomerID', 'Timestamp']).reset_index(drop=True)

# # 1. 基础时间特征
# full['txn_hour'] = full['Timestamp'].dt.hour
# full['txn_dow'] = full['Timestamp'].dt.dayofweek
# full['txn_is_weekend'] = (full['txn_dow'] >= 5).astype(int)
# full['txn_month'] = full['Timestamp'].dt.month
# full['txn_day'] = full['Timestamp'].dt.day
# full['is_night'] = ((full['txn_hour'] < 6) | (full['txn_hour'] >= 22)).astype(int)
# full['days_since_login'] = (full['Timestamp'] - full['LastLogin']).dt.days

# # 2. 时间窗口特征
# def rolling_txn_count(s, days):
#     t = s.values
#     delta = np.timedelta64(days, 'D')
#     out = np.zeros(len(t), dtype=int)
#     for i in range(len(t)):
#         out[i] = ((t < t[i]) & (t >= t[i] - delta)).sum()
#     return out

# full['txn_order'] = full.groupby('CustomerID').cumcount() + 1
# full['prev_gap_hours'] = full.groupby('CustomerID')['Timestamp'].diff().dt.total_seconds() / 3600
# full['prev_gap_hours'] = full['prev_gap_hours'].fillna(0)
# full['txn_last7d'] = full.groupby('CustomerID')['Timestamp'].transform(lambda s: rolling_txn_count(s, 7))
# full['txn_last30d'] = full.groupby('CustomerID')['Timestamp'].transform(lambda s: rolling_txn_count(s, 30))

# # 3. 金额-账户交互
# full['amount_ratio'] = full['Amount'] / (full['AccountBalance'] + 1)
# full['amount_over_balance'] = (full['Amount'] > full['AccountBalance']).astype(int)

# # 4. 交叉特征：SuspiciousFlag × 金额/大额/余额比/深夜
# large_thr = full['Amount'].quantile(0.9)                       # "大额"=金额 90 分位
# full['is_large'] = (full['Amount'] >= large_thr).astype(int)
# full['susp_amount'] = full['SuspiciousFlag'] * full['Amount']            # 可疑客户 × 金额
# full['susp_large'] = full['SuspiciousFlag'] * full['is_large']           # 可疑客户 × 大额交易
# full['susp_amount_ratio'] = full['SuspiciousFlag'] * full['amount_ratio']  # 可疑客户 × 占余额比
# full['susp_over_balance'] = full['SuspiciousFlag'] * full['amount_over_balance']  # 可疑客户 × 透支
# full['susp_night'] = full['SuspiciousFlag'] * full['is_night']           # 可疑客户 × 深夜交易

# # 5. 年龄分桶
# full['age_bin'] = pd.cut(full['Age'], bins=[0, 20, 30, 40, 50, 60, 120],
#                          labels=[0, 1, 2, 3, 4, 5]).astype(float)

# # 6. 客户聚合（不含标签）
# g = full.groupby('CustomerID')
# cust = pd.DataFrame({
#     'cust_txn_cnt': g.size(),
#     'cust_amt_mean': g['Amount'].mean(),
#     'cust_amt_max': g['Amount'].max(),
#     'cust_amt_std': g['Amount'].std().fillna(0),
#     'cust_n_merchant': g['MerchantID'].nunique(),
#     'cust_n_category': g['Category'].nunique(),
#     'cust_span_days': (g['Timestamp'].max() - g['Timestamp'].min()).dt.days,
# }).reset_index()
# full = full.merge(cust, on='CustomerID', how='left')
# full['cust_daily_freq'] = full['cust_txn_cnt'] / (full['cust_span_days'] + 1)

# # 7. 商户聚合（不含标签）
# g2 = full.groupby('MerchantID')
# merch = pd.DataFrame({
#     'merch_txn_cnt': g2.size(),
#     'merch_amt_mean': g2['Amount'].mean(),
#     'merch_n_customer': g2['CustomerID'].nunique(),
# }).reset_index()
# full = full.merge(merch, on='MerchantID', how='left')

# 8. OOF 目标编码
def oof_target_encode(full, cat_col, target_col=TARGET, n_splits=5, smoothing=20, seed=42):
    tr_mask = full[target_col].notna()
    tr = full[tr_mask].reset_index(drop=True)
    te = full[~tr_mask].reset_index(drop=True)
    y = tr[target_col].astype(int).values
    global_mean = y.mean()

    def encode(ids, m_map, c_map):
        m = pd.Series(ids).map(c_map).fillna(0).values
        mean = pd.Series(ids).map(m_map).fillna(global_mean).values
        return (m * mean + smoothing * global_mean) / (m + smoothing)

    st = tr.groupby(cat_col)[target_col].agg(['mean', 'count'])
    te_enc = encode(te[cat_col].astype(str).values, st['mean'], st['count'])
    tr_enc = np.zeros(len(tr))
    for tr_idx, va_idx in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(tr, y):
        fs = tr.iloc[tr_idx].groupby(cat_col)[target_col].agg(['mean', 'count'])
        tr_enc[va_idx] = encode(tr.iloc[va_idx][cat_col].astype(str).values, fs['mean'], fs['count'])
    out = np.full(len(full), np.nan)
    out[tr_mask.values] = tr_enc
    out[~tr_mask.values] = te_enc
    full[cat_col + '_te'] = out
    return full

full = oof_target_encode(full, 'CustomerID')
full = oof_target_encode(full, 'MerchantID')

# 9. drop 原始时间/高基数列
full = full.drop(columns=['Timestamp', 'LastLogin', 'Name', 'MerchantName'])

# 10. 低基数 one-hot
onehot_cols = ['Sex', 'Address', 'Location', 'Category']
full = pd.get_dummies(full, columns=onehot_cols, prefix=onehot_cols)

# ================= 清洗 =================
for c in full.columns:
    if c in (TARGET, ID_COL) or full[c].dtype != 'object':
        continue
    nonnull = full[c].notna()
    s = full[c].astype(str).str.strip().str.replace(r'[,$%\s]', '', regex=True)
    s = s.replace(['', 'nan', 'None', 'NULL', '?', '.'], np.nan)
    conv = pd.to_numeric(s, errors='coerce')
    if nonnull.sum() > 0 and (conv.notna() & nonnull).sum() / nonnull.sum() >= 0.9:
        full[c] = conv

for c in full.columns:
    if full[c].dtype == 'object' and c != ID_COL:
        full[c] = full[c].astype(str).str.strip()
        full.loc[full[c].isin(['', 'nan', 'NaN', 'None', 'NULL', '?', '.']), c] = np.nan

num_cols = [c for c in full.columns
            if full[c].dtype != 'object' and c not in (TARGET, ID_COL)]
for c in num_cols:
    full[c] = pd.to_numeric(full[c], errors='coerce')
    full[c] = full[c].fillna(full[c].median())

cat_cols = [c for c in full.columns
            if full[c].dtype == 'object' and c not in (TARGET, ID_COL)]
for c in cat_cols:
    full[c] = full[c].fillna(full[c].mode()[0] if not full[c].mode().empty else 'Unknown')

for c in cat_cols:
    le = LabelEncoder()
    le.fit(list(full[c].dropna().unique()))
    full[c] = le.transform(full[c].astype(str))

# ================= 拆回训练/测试 =================
train_df = full[full[TARGET].notna()].copy()
test_df = full[full[TARGET].isna()].copy()
drop_cols = [ID_COL, TARGET]
feat_cols = [c for c in train_df.columns if c not in drop_cols]
X = train_df[feat_cols]
y = train_df[TARGET].astype(int).values
X_test = test_df[feat_cols]
test_ids = test_df[ID_COL]

print('\n特征数:', len(feat_cols), ' 训练集:', X.shape, ' 测试集:', X_test.shape)
print('正类比例:', round(y.mean(), 4))

# ================= 模型（加早停 + 更细调参） =================
scale = (y == 0).sum() / max((y == 1).sum(), 1)
ES_ROUNDS = 100
MAX_EST = 3000
models = {
    'LGBM': LGBMClassifier(
        learning_rate=0.02, num_leaves=127, min_child_samples=20,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1),
    'XGB': XGBClassifier(
        learning_rate=0.02, max_depth=7, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale, random_state=42, n_jobs=-1, verbosity=0),
    'RF': RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def fit_es(m2, X_tr, y_tr, X_es, y_es):
    """带早停的 fit（只对 LGBM/XGB；RF 不走这里）。"""
    m2.set_params(n_estimators=MAX_EST)
    if isinstance(m2, LGBMClassifier):
        m2.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric='auc',
               callbacks=[lgbm_es(ES_ROUNDS, verbose=False)])
    else:  # XGB
        m2.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric='auc',
               early_stopping_rounds=ES_ROUNDS, verbose=False)
    return m2


def oof_proba(m):
    oof = np.zeros(len(y))
    test_p = np.zeros(len(X_test))
    for tr, va in skf.split(X, y):
        m2 = m.__class__(**m.get_params())
        if isinstance(m2, (LGBMClassifier, XGBClassifier)):
            # 每折在 tr 内再切 15% 做早停验证，va 完全不参与
            X_tr, X_es, y_tr, y_es = train_test_split(
                X.iloc[tr], y[tr], test_size=0.15, stratify=y[tr], random_state=42)
            m2 = fit_es(m2, X_tr, y_tr, X_es, y_es)
        else:
            m2.fit(X.iloc[tr], y[tr])
        oof[va] = m2.predict_proba(X.iloc[va])[:, 1]
        test_p += m2.predict_proba(X_test)[:, 1] / 5
    return oof, test_p


best_name, best_oof, best_test, best_f1 = None, None, None, -1
for name, m in models.items():
    try:
        oof, test_p = oof_proba(m)
        f = f1_score(y, (oof >= 0.5).astype(int))
        auc = roc_auc_score(y, oof)
        print(f'{name:5s}  OOF F1@0.5={f:.4f}  AUC={auc:.4f}')
        if f > best_f1:
            best_name, best_oof, best_test, best_f1 = name, oof, test_p, f
    except Exception as e:
        print(f'{name:5s}  失败 {type(e).__name__}: {str(e)[:80]}')
print(f'选中模型: {best_name}   OOF F1@0.5={best_f1:.4f}')

# ================= 阈值 + 输出 =================
best_thr, best_f1_thr = 0.5, 0
for t in np.arange(0.05, 0.95, 0.005):
    f = f1_score(y, (best_oof >= t).astype(int))
    if f > best_f1_thr:
        best_f1_thr, best_thr = f, t
print(f'最优阈值={best_thr:.3f}  OOF F1={best_f1_thr:.4f}')

pred = (best_test >= best_thr).astype(int)
result = pd.DataFrame({ID_COL: test_ids, 'Predicted_Results': pred})
result.to_csv('202412_results.csv', index=False)
print('预测分布:\n', result['Predicted_Results'].value_counts())

# ================= 特征重要性（最佳模型在全量训练集上训一次） =================
print(f'\n===== 特征重要性 Top 20（{best_name}）=====')
final = models[best_name].__class__(**models[best_name].get_params())
if isinstance(final, (LGBMClassifier, XGBClassifier)):
    X_tr2, X_es2, y_tr2, y_es2 = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    final = fit_es(final, X_tr2, y_tr2, X_es2, y_es2)
    if isinstance(final, LGBMClassifier):
        imp = final.booster_.feature_importance(importance_type='gain')
        n_iter = final.best_iteration_
    else:
        imp = final.feature_importances_          # XGB 默认 gain
        n_iter = final.best_iteration
    print(f'(早停在第 {n_iter} 棵树)')
else:
    final.fit(X, y)
    imp = final.feature_importances_
imp_df = pd.DataFrame({'feature': feat_cols, 'importance': imp}).sort_values(
    'importance', ascending=False).reset_index(drop=True)
print(imp_df.head(20).to_string(index=False))
