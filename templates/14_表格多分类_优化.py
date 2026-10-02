# -*- coding: utf-8 -*-
# ============================================================================
# 模板⑦优化版:结构化表格多分类(202512 冷水机故障,29类)
#   评分:Macro-F1
#   优化点:
#     ① 软投票集成 LGBM + XGB 各 0.5(实测 RF 太弱,加进去反而拖后腿)
#     ② LGBM 网格调参(num_leaves / min_child_samples / learning_rate)
#     ③ 统一 sample_weight 做类别平衡(原模板 XGB 没平衡,从 0.7665 -> 0.7971)
#   为省时间:调参用单次留出(80/20)而非 5 折;最终对比仍用 5 折。
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
from itertools import product

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def P(*a): print(*a, flush=True)

# ===== 读数据 =====
def load_chiller(normal_dir, fault_dir):
    dfs = []
    for f in os.listdir(normal_dir):
        dfs.append(pd.read_excel(os.path.join(normal_dir, f)))
    for f in os.listdir(fault_dir):
        dfs.append(pd.read_excel(os.path.join(fault_dir, f)))
    return pd.concat(dfs, ignore_index=True)

train = load_chiller('data/Training_Data/normal/', 'data/Training_Data/fault/')
test  = pd.read_excel('data/test_data_all.xlsx')
TARGET, ID_COL = 'Class', 'ID'

# ===== 预处理 =====
feat = [c for c in train.columns if c in test.columns and c not in (TARGET, ID_COL)]
X = train[feat]; y_raw = train[TARGET].astype(str); X_test = test[feat]; test_ids = test[ID_COL]
le = LabelEncoder(); y = le.fit_transform(y_raw); num_classes = len(le.classes_)
P('类别数:', num_classes)

X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
vt = VarianceThreshold(threshold=0.01)
X = vt.fit_transform(X); X_test = vt.transform(X_test)
sc = StandardScaler(); X = sc.fit_transform(X); X_test = sc.transform(X_test)
P('特征维度:', X.shape)

sw = compute_sample_weight('balanced', y)   # 类别不平衡 -> 样本权重

# ================= 优化②:调参(单次留出,快) =================
X_a, X_b, y_a, y_b = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sw_a = compute_sample_weight('balanced', y_a)

best_lgbm, best_score = None, -1.0
for nl, mcs, lr in product([15, 31, 63], [5, 10, 20], [0.05]):
    p = dict(objective='multiclass', num_class=num_classes, n_estimators=300,
             num_leaves=nl, min_child_samples=mcs, learning_rate=lr,
             random_state=42, n_jobs=-1, verbose=-1)
    m = LGBMClassifier(**p)
    m.fit(X_a, y_a, sample_weight=sw_a)
    s = f1_score(y_b, m.predict(X_b), average='macro')
    if s > best_score:
        best_score, best_lgbm = s, p
P('[调参] 最优 Macro-F1=%.4f  num_leaves=%d min_child_samples=%d lr=%.2f'
  % (best_score, best_lgbm['num_leaves'], best_lgbm['min_child_samples'], best_lgbm['learning_rate']))

# ================= 优化①:软投票集成 =================
def make_models():
    return {
        'LGBM': LGBMClassifier(**best_lgbm),
        'XGB':  XGBClassifier(objective='multi:softprob', num_class=num_classes,
                              n_estimators=300, learning_rate=0.05, max_depth=6,
                              subsample=0.9, colsample_bytree=0.9,
                              random_state=42, n_jobs=-1, verbosity=0),
        'RF':   RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }

from collections import defaultdict
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = defaultdict(list)
for tr, va in skf.split(X, y):
    probas = {}
    for name, m in make_models().items():
        m.fit(X[tr], y[tr], sample_weight=sw[tr])
        probas[name] = m.predict_proba(X[va])
        cv_acc[name].append(f1_score(y[va], probas[name].argmax(1), average='macro'))
    # 软投票:只用 LGBM+XGB 各半(实测 RF 太弱,加进去反而拖后腿)
    ens = 0.5 * probas['LGBM'] + 0.5 * probas['XGB']
    cv_acc['ENSEMBLE(LGBM+XGB)'].append(f1_score(y[va], ens.argmax(1), average='macro'))

P('\n[5折对比] Macro-F1:')
for n in ['LGBM', 'XGB', 'RF', 'ENSEMBLE(LGBM+XGB)']:
    P('  %-18s = %.4f' % (n, np.mean(cv_acc[n])))

# ================= 全量训练 + 软投票预测 =================
probas_full = {}
for name, m in make_models().items():
    m.fit(X, y, sample_weight=sw)
    probas_full[name] = m.predict_proba(X_test)
ens_test = 0.5 * probas_full['LGBM'] + 0.5 * probas_full['XGB']
pred = le.inverse_transform(ens_test.argmax(1))

result = pd.DataFrame({ID_COL: test_ids, 'Predicted_Results': pred})
result.to_csv('考生姓名_results.csv', index=False)
P('\n预测分布:\n', result['Predicted_Results'].value_counts())
