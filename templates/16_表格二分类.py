# ============================================================================
# 模板①：结构化表格数据 —— 二分类（最常考，必背）
# 适用：201812产品营销 / 201906流失 / 201912借贷 / 202412欺诈 / 202603流失 / 2026.06电商
# 评分：正类 F1（流失/购买/欺诈/Revenue=Yes 的 F1），越大越好
# 只看"===== 改这里 =====" 标注的几处，其余照抄即可
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)

# ===== 改这里 1：数据路径 + 目标列名 + ID 列名 =====
train = pd.read_csv('ecommerce_train.csv', encoding='utf-8-sig', na_values=['?', '.'])   # Excel 用 pd.read_excel
test  = pd.read_csv('ecommerce_test.csv',  encoding='utf-8-sig', na_values=['?', '.'])
TARGET = 'Revenue'      # 目标列名
ID_COL = 'ID'         # 提交用的 ID 列名
# ====================================================

# ---------- 1. 目标标签统一转成 0/1 ----------
label_map = {'True': 1, 'False': 0}     # 本数据 Revenue 是 True/False
train[TARGET] = train[TARGET].astype(str).str.strip().map(label_map)
train = train.dropna(subset=[TARGET]).copy()   # 删掉没有标签的行
train[TARGET] = train[TARGET].astype(int)

# ---------- 2. 合并训练测试，统一做清洗/编码（最后再拆开）----------
test[TARGET] = np.nan
full = pd.concat([train, test], axis=0, ignore_index=True)

# 2.1 字符串类别列：去空格，空白/占位当缺失
for c in full.columns:
    if full[c].dtype == 'object' and c != ID_COL:
        full[c] = full[c].astype(str).str.strip()
        full.loc[full[c].isin(['', 'nan', 'NaN', 'None', 'NULL', '?', '.']), c] = np.nan

# 2.2 数值列：转数值（空白串会变 NaN），用中位数填充
num_cols = [c for c in full.columns
            if full[c].dtype != 'object' and c not in (TARGET, ID_COL)]
for c in num_cols:
    full[c] = pd.to_numeric(full[c], errors='coerce')
    full[c] = full[c].fillna(full[c].median())

# 2.3 类别列：缺失填众数/Unknown
cat_cols = [c for c in full.columns
            if full[c].dtype == 'object' and c not in (TARGET, ID_COL)]
for c in cat_cols:
    full[c] = full[c].fillna(full[c].mode()[0] if not full[c].mode().empty else 'Unknown')

# ---------- 3. 特征工程（按题目字段发挥，这里给通用招式）----------
# 3.0 月份顺序编码（必须放在 3.1 之前！LabelEncoder 会把 Month 转成数字）
month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'June':6,'Jul':7,
             'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
full['Month_num'] = full['Month'].map(month_map).astype('float')

# 3.1 类别编码：训练+测试一起 fit，保证编码一致
for c in cat_cols:
    le = LabelEncoder()
    le.fit(list(full[c].dropna().unique()))
    full[c] = le.transform(full[c].astype(str))

# 3.2 常用衍生特征示例（数值列之间做比率/分桶，视情况用）
# full['A_per_B'] = full['colA'] / (full['colB'] + 1)
# full['age_bin'] = pd.cut(full['age'], bins=[0,20,40,60,120], labels=[0,1,2,3]).astype(float)
full['Total_Duration'] = (full['Admin_Duration'] + full['Info_Duration']
                          + full['Product_Duration'])
full['Total_Pages'] = (full['Administrative'] + full['Informational']
                       + full['ProductRelated'])

# --- 3.2 平均每页停留时长(用户粘性) ---
full['Avg_Duration_per_Page'] = full['Total_Duration'] / (full['Total_Pages'] + 1)
full['Avg_Product_Duration']  = full['Product_Duration'] / (full['ProductRelated'] + 1)
full['Avg_Admin_Duration']    = full['Admin_Duration']   / (full['Administrative'] + 1)
full['Avg_Info_Duration']     = full['Info_Duration']    / (full['Informational'] + 1)

# --- 3.3 各类页面占比(用户行为结构) ---
full['Product_Pages_Ratio'] = full['ProductRelated'] / (full['Total_Pages'] + 1)
full['Admin_Pages_Ratio']   = full['Administrative'] / (full['Total_Pages'] + 1)
full['Info_Pages_Ratio']    = full['Informational']  / (full['Total_Pages'] + 1)

# --- 3.4 各类页面时长占比 ---
full['Product_Duration_Ratio'] = full['Product_Duration'] / (full['Total_Duration'] + 1)
full['Admin_Duration_Ratio']   = full['Admin_Duration']   / (full['Total_Duration'] + 1)

# --- 3.5 跳出/离开相关组合 ---
full['Bounce_Exit_Mean'] = (full['BounceRates'] + full['ExitRates']) / 2
full['High_Bounce']      = (full['BounceRates'] > 0.2).astype(int)
full['Low_Exit']         = (full['ExitRates'] < 0.05).astype(int)

# --- 3.6 PageValues 相关 ---
full['PageValues_Log']  = np.log1p(full['PageValues'])
full['High_PageValues'] = (full['PageValues'] > full['PageValues'].quantile(0.75)).astype(int)

# --- 3.7 特节庆/周末交互 ---
full['SpecialDay_High'] = (full['SpecialDay'] > 0.5).astype(int)

# ---------- 4. 拆回训练/测试 ----------
train_df = full[full[TARGET].notna()].copy()
test_df  = full[full[TARGET].isna()].copy()

drop_cols = [ID_COL, TARGET]
feat_cols = [c for c in train_df.columns if c not in drop_cols]
X = train_df[feat_cols]
y = train_df[TARGET].astype(int).values
X_test = test_df[feat_cols]
test_ids = test_df[ID_COL]

print('特征数:', len(feat_cols), ' 训练集:', X.shape, ' 测试集:', X_test.shape)
print('正类比例:', y.mean().round(4))

# ---------- 5. 多模型 5 折对比 ----------
scale = (y == 0).sum() / max((y == 1).sum(), 1)   # 不平衡比例
models = {
    'LGBM': LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                           scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1),
    'XGB':  XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                          scale_pos_weight=scale, random_state=42, n_jobs=-1,
                          eval_metric='logloss', verbosity=0),
    'RF':   RandomForestClassifier(n_estimators=500, max_depth=15,
                                   class_weight='balanced', random_state=42, n_jobs=-1),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in models.items():
    f1s, aucs = [], []
    for tr, va in skf.split(X, y):
        m2 = m.__class__(**m.get_params())
        m2.fit(X.iloc[tr], y[tr])
        p = m2.predict_proba(X.iloc[va])[:, 1]
        f1s.append(f1_score(y[va], (p >= 0.5).astype(int)))
        aucs.append(roc_auc_score(y[va], p))
    print(f'{name:5s}  F1@0.5={np.mean(f1s):.4f}  AUC={np.mean(aucs):.4f}')

# ---------- 6. 用 LightGBM 做 OOF 概率 + 阈值优化（评分是正类F1时）----------
BEST = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=63,
                      scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1)
oof = np.zeros(len(y))
test_proba = np.zeros(len(X_test))
for tr, va in skf.split(X, y):
    m = BEST.__class__(**BEST.get_params())
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
          callbacks=[__import__('lightgbm').early_stopping(100, verbose=False)])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    test_proba += m.predict_proba(X_test)[:, 1] / 5

# 搜 F1 最大的阈值
best_thr, best_f1 = 0.5, 0
for t in np.arange(0.05, 0.95, 0.005):
    f = f1_score(y, (oof >= t).astype(int))
    if f > best_f1:
        best_f1, best_thr = f, t
print(f'最优阈值={best_thr:.3f}  OOF F1={best_f1:.4f}')

# ---------- 7. 输出提交文件 ----------
pred = (test_proba >= best_thr).astype(int)

# ===== 改这里 2：把 0/1 反编码回题目要求的字符串 =====
inv_map = {1: 'True', 0: 'False'}       # 本数据 Revenue 是 True/False
# ====================================================

result = pd.DataFrame({
    ID_COL: test_ids,
    'Predicted_Results': [inv_map.get(x, x) for x in pred],   # 列名按题目改
})
result.to_csv('考生姓名_results.csv', index=False)
print(result.head())
print('预测分布:\n', result['Predicted_Results'].value_counts())


# ============================================================================
# 变体 A：评分是"利润(成本矩阵)"时（201812 产品营销）
# 思路：按概率从高到低排序，逐步把样本判为正类，找累计利润最大的那个概率阈值
# 利润 = TP*正类利润 - FP*错判成本（题给：卖出+1500，错投-500）
# ============================================================================
def profit_threshold(oof, y, gain=1500, cost=500):
    order = np.argsort(-oof)          # 概率从大到小
    y_sorted = y[order]
    profit, best_p, best_profit = 0, oof[order][0], -1e18
    for i, yi in enumerate(y_sorted):
        profit += gain if yi == 1 else -cost
        if profit > best_profit:
            best_profit, best_p = profit, oof[order][i]
    return best_p, best_profit
# 用法：用上面的 oof/y 算阈值，再对 test_proba 套阈值，输出 0/1 即可
