# ============================================================================
# 模板⑦：结构化表格数据 —— 多分类（一个样本属于 N 类中的一个）
# 适用：202512 冷水机故障(29类: normal + cf1..ro4)
# 评分：Macro-F1
# 核心：LabelEncoder 编码类别 -> 训练多分类器 -> inverse_transform 反编码回类名
# ⚠️ 类名大小写必须完全一致(normal/cf1/cf2/.../ro4)，用 inverse_transform 最保险
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# ===== 改这里 1：读数据 =====
# 202512 是多个 excel 按 normal/fault 文件夹分开放，先合并：
def load_chiller(normal_dir, fault_dir):
    dfs = []
    for f in os.listdir(normal_dir):
        dfs.append(pd.read_excel(os.path.join(normal_dir, f)))
    for f in os.listdir(fault_dir):
        dfs.append(pd.read_excel(os.path.join(fault_dir, f)))
    return pd.concat(dfs, ignore_index=True)

train = load_chiller('data/Training_Data/normal/', 'data/Training_Data/fault/')
test  = pd.read_excel('data/test_data_all.xlsx')

TARGET = 'Class'     # 目标列名（有的题叫 'class'，注意大小写）
ID_COL = 'ID'
# ====================================================

# ---------- 1. 特征只取训练/测试两边都有的列（202512 里 train 多了 Heat Balance，剔除掉） ----------
feat = [c for c in train.columns
        if c in test.columns and c not in (TARGET, ID_COL)]
X = train[feat]
y_raw = train[TARGET].astype(str)
X_test = test[feat]
test_ids = test[ID_COL]

# ---------- 2. 编码目标 ----------
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)
print('类别数:', num_classes, ' 类名:', list(le.classes_))

# ---------- 3. 数值预处理：中位数填充 + 方差过滤 + 标准化 ----------
X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

vt = VarianceThreshold(threshold=0.01)
X = vt.fit_transform(X)
X_test = vt.transform(X_test)
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)
print('特征维度:', X.shape)

# ---------- 4. 多模型 5 折对比（Macro-F1） ----------
models = {
    'LGBM': LGBMClassifier(objective='multiclass', num_class=num_classes,
                           n_estimators=300, learning_rate=0.05, num_leaves=31,
                           class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
    'XGB':  XGBClassifier(objective='multi:softprob', num_class=num_classes,
                          n_estimators=300, learning_rate=0.05, max_depth=6,
                          random_state=42, n_jobs=-1, verbosity=0),
    'RF':   RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                   random_state=42, n_jobs=-1),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in models.items():
    scores = []
    for tr, va in skf.split(X, y):
        m2 = m.__class__(**m.get_params())
        m2.fit(X[tr], y[tr])
        scores.append(f1_score(y[va], m2.predict(X[va]), average='macro'))
    print(f'{name:5s}  Macro-F1={np.mean(scores):.4f}')

# ---------- 5. 全量训练 + 预测 + 反编码 ----------
best = LGBMClassifier(objective='multiclass', num_class=num_classes,
                      n_estimators=300, learning_rate=0.05, num_leaves=31,
                      class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
best.fit(X, y)
pred = le.inverse_transform(best.predict(X_test))   # 自动还原成类名(normal/cf1...)

result = pd.DataFrame({ID_COL: test_ids, 'Predicted_Results': pred})
result.to_csv('考生姓名_results.csv', index=False)
print(result.head())
print('预测分布:\n', result['Predicted_Results'].value_counts())
