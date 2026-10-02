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
train = pd.read_csv('telecom_churn_train.csv', encoding='utf-8-sig', na_values=['?', '.'])   # Excel 用 pd.read_excel
test  = pd.read_csv('telecom_churn_test.csv',  encoding='utf-8-sig', na_values=['?', '.'])
TARGET = 'Churn'      # 目标列名
ID_COL = 'id'         # 提交用的 ID 列名
# ====================================================

# ---------- 1. 目标标签统一转成 0/1 ----------
# 题目可能给 Yes/No、True/False、1/0，统一映射；把字典按题目实际写
label_map = {'Yes': 1, 'No': 0}          # 例：202603 是 Yes/No
# label_map = {'TRUE': 1, 'FALSE': 0}    # 例：2026.06 电商
# label_map = {'True': 1, 'False': 0}    # 例：201906 流失
train[TARGET] = train[TARGET].astype(str).str.strip().map(label_map)
train = train.dropna(subset=[TARGET]).copy()   # 删掉没有标签的行(半监督题的少量无标签行)
train[TARGET] = train[TARGET].astype(int)

# ---------- 2. 合并训练测试，统一做清洗/编码（最后再拆开）----------
test[TARGET] = np.nan
full = pd.concat([train, test], axis=0, ignore_index=True)

# 2.1 抢救"被读成文本的数值列"（脏数据题关键，如 201912 借贷）
#     金额/年限等列常混进 $、逗号、空格或脏值，被 pandas 读成 object；
#     不抢救就会被当类别列做 LabelEncoder，白白丢掉数值信息。
for c in full.columns:
    if c in (TARGET, ID_COL) or full[c].dtype != 'object':
        continue
    nonnull = full[c].notna()
    s = full[c].astype(str).str.strip().str.replace(r'[,$%\s]', '', regex=True)
    s = s.replace(['', 'nan', 'None', 'NULL', '?', '.'], np.nan)
    conv = pd.to_numeric(s, errors='coerce')
    # 在原非缺失值里，九成以上能转成数字 → 就按数值列处理（阈值 0.9 防误判）
    if nonnull.sum() > 0 and (conv.notna() & nonnull).sum() / nonnull.sum() >= 0.9:
        full[c] = conv

# 2.2 字符串类别列：去空格，空白/占位当缺失
for c in full.columns:
    if full[c].dtype == 'object' and c != ID_COL:
        full[c] = full[c].astype(str).str.strip()
        full.loc[full[c].isin(['', 'nan', 'NaN', 'None', 'NULL', '?', '.']), c] = np.nan

# 2.3 数值列：转数值（空白串会变 NaN），用中位数填充
num_cols = [c for c in full.columns
            if full[c].dtype != 'object' and c not in (TARGET, ID_COL)]
for c in num_cols:
    full[c] = pd.to_numeric(full[c], errors='coerce')
    full[c] = full[c].fillna(full[c].median())

# 2.4 类别列：缺失填众数/Unknown
cat_cols = [c for c in full.columns
            if full[c].dtype == 'object' and c not in (TARGET, ID_COL)]
for c in cat_cols:
    full[c] = full[c].fillna(full[c].mode()[0] if not full[c].mode().empty else 'Unknown')

# ---------- 3. 特征工程（按题目字段发挥，这里给通用招式）----------
# 3.05 业务衍生特征(领域知识驱动,库替代不了;必须在 3.1 编码前,因为要用 Yes/No 原始字符串)
# ① 消费结构:总消费/月费。越大越像"一次结清/长期",越小越像"月付观望"
full['TotalToMonthlyRatio'] = full['TotalCharges'] / (full['MonthlyCharges'] + 1)

# ② 附加服务数量:6 个增值服务里 Yes 的个数(服务越多越稳定,越不易流失)
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
full['NumServices'] = (full[service_cols] == 'Yes').astype(int).sum(axis=1)

# ③ 是否光纤(光纤用户流失率显著更高,是 Telco 最强单特征之一)
full['IsFiber'] = (full['InternetService'] == 'Fiber optic').astype(int)

# ④ 是否自动支付(自动续费,流失率低)
full['IsAutoPay'] = full['PaymentMethod'].astype(str).str.contains('automatic', case=False).astype(int)

# ⑤ 合同年限有序化(月付<一年<两年;3.1 会按字母序乱编,这里给正确顺序)
full['ContractLen'] = full['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2}).fillna(-1)

# 3.1 类别编码：训练+测试一起 fit，保证编码一致
for c in cat_cols:
    le = LabelEncoder()
    le.fit(list(full[c].dropna().unique()))
    full[c] = le.transform(full[c].astype(str))

# 3.2 常用衍生特征示例（数值列之间做比率/分桶，视情况用）
# full['A_per_B'] = full['colA'] / (full['colB'] + 1)
# full['age_bin'] = pd.cut(full['age'], bins=[0,20,40,60,120], labels=[0,1,2,3]).astype(float)

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

# ---------- 5. 多模型 5 折 OOF 对比，按 F1@0.5 选最优（对齐模板②：手写 OOF，绕开 cross_val_predict bug）----------
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


def oof_proba(m):
    """手写 out-of-fold 概率 + 测试集 5 折平均概率。"""
    oof = np.zeros(len(y))
    test_p = np.zeros(len(X_test))
    for tr, va in skf.split(X, y):
        m2 = m.__class__(**m.get_params())
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
        print(f'{name:5s}  失败 {type(e).__name__}: {str(e)[:60]}')
print(f'选中模型: {best_name}   OOF F1@0.5={best_f1:.4f}')

# ---------- 6. 用最优模型的 OOF 概率搜阈值（评分是正类F1时）----------
best_thr, best_f1_thr = 0.5, 0
for t in np.arange(0.05, 0.95, 0.005):
    f = f1_score(y, (best_oof >= t).astype(int))
    if f > best_f1_thr:
        best_f1_thr, best_thr = f, t
print(f'最优阈值={best_thr:.3f}  OOF F1={best_f1_thr:.4f}')

# ---------- 7. 输出提交文件 ----------
pred = (best_test >= best_thr).astype(int)

# ===== 改这里 2：把 0/1 反编码回题目要求的字符串 =====
inv_map = {1: 'Yes', 0: 'No'}           # 例：202603
# inv_map = {1: 'TRUE', 0: 'FALSE'}     # 例：2026.06 电商
# inv_map = {1: 'True', 0: 'False'}     # 例：201906
# inv_map = {1: 'H', 0: 'L'}            # 例：202103(文本题，此处仅示意)
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
