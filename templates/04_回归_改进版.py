# ============================================================================
# 练习2209「共享住宿日租价格预测」改进版
# 评分：RMSE（越小越好，按排名给分：前5%=30 / 前10%=24 / 前20%=18 / 前40%=12 / 前80%=6 / 后20%=0）
#
# 相比模板④的三个修正：
#   1. 【致命·格式】提交列名写成了 predicted_rank，但题目要求的是
#      predicted_rrental_price_USD —— 注意是双 r 的 "rrental"（题目原文拼写错误）。
#      列名错了 = 格式错误 = 直接 0 分，跟模型好坏无关。
#   2. 【模型】amenities 是「设施列表」（4667 行 3666 个唯一值）。
#      模板④把它当普通文本列做 LabelEncoder → 每行一个几乎唯一的编号 → 白占一个特征还过拟合。
#      这里拆成 multi-hot（高频设施各一个 0/1 特征）+ 设施数量。
#   3. 目标高度右偏（max=10000，中位数=175），加 log1p 变换通常能再压一点 RMSE。
#
# 用 test-ans.xlsx 自带的真实答案直接算 TEST RMSE 验证。
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping

TRAIN = 'data/training.xlsx'
TEST  = 'data/test-ans.xlsx'          # 带真实答案，用来算 TEST RMSE
TARGET = 'rental_price_USD'
ID_COL = 'room_id'
OUT_COL = 'predicted_rrental_price_USD'   # ★ 题目原文双 r，千万别改
USE_LOG = True                            # log1p 变换（右偏价格通常更稳）

train = pd.read_excel(TRAIN)
test  = pd.read_excel(TEST)
test_true = test[TARGET].values
test = test.drop(columns=[TARGET])        # 预测时先拿掉答案列


# ---------- 1. 合并清洗（同模板①），amenities 单独处理 ----------
test[TARGET] = np.nan
full = pd.concat([train, test], axis=0, ignore_index=True)


def parse_amenities(s):
    s = str(s).strip().replace('{', '').replace('}', '')
    return [x.strip() for x in s.split(',') if x.strip()]


# amenities → 数量 + 高频设施 multi-hot（train+test 一起统计）
amen_lists = full['amenities'].apply(parse_amenities)
full['len_amenities'] = amen_lists.apply(len)
cnt = Counter()
amen_lists.apply(cnt.update)
for k, _ in cnt.most_common(60):
    full['am_' + k] = amen_lists.apply(lambda lst, k=k: 1 if k in lst else 0)
full = full.drop(columns=['amenities'])

# 数值列：抢救被读成文本的数值 + 中位数填充（同模板④）
for c in full.columns:
    if c in (TARGET, ID_COL):
        continue
    if full[c].dtype != 'object':
        full[c] = pd.to_numeric(full[c], errors='coerce')
        full[c] = full[c].fillna(full[c].median())
    else:
        full[c] = full[c].astype(str).str.strip()
        full.loc[full[c].isin(['', 'nan', 'NaN', 'None', 'NULL', '?', '.']), c] = np.nan
        full[c] = full[c].fillna(full[c].mode()[0] if not full[c].mode().empty else 'Unknown')
        full[c] = LabelEncoder().fit_transform(full[c].astype(str))

train_df = full[full[TARGET].notna()].copy()
test_df  = full[full[TARGET].isna()].copy()
feat_cols = [c for c in train_df.columns if c not in (TARGET, ID_COL)]   # ★ 排除 room_id
X = train_df[feat_cols]
y = train_df[TARGET].astype(float).values
X_test = test_df[feat_cols]
test_ids = test_df[ID_COL].values
print('特征数:', len(feat_cols), ' X:', X.shape, ' X_test:', X_test.shape, flush=True)

yy = np.log1p(y) if USE_LOG else y


# ---------- 2. K 折 + 早停 ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(yy))
test_pred = np.zeros(len(X_test))
best_iters = []
for tr_i, va in kf.split(X):
    m = LGBMRegressor(objective='regression_l1', n_estimators=4000, learning_rate=0.02,
                      num_leaves=31, colsample_bytree=0.8, subsample=0.8,
                      reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr_i], yy[tr_i], eval_set=[(X.iloc[va], yy[va])],
          callbacks=[early_stopping(100, verbose=False)])
    oof[va] = m.predict(X.iloc[va])
    test_pred += m.predict(X_test) / 5
    best_iters.append(m.best_iteration_)

if USE_LOG:
    oof, test_pred = np.expm1(oof), np.expm1(test_pred)

print(f'OOF_RMSE={np.sqrt(mean_squared_error(y, oof)):.2f}   '
      f'TEST_RMSE={np.sqrt(mean_squared_error(test_true, test_pred)):.2f}   '
      f'平均最佳轮数={int(np.mean(best_iters))}', flush=True)


# ---------- 3. 全量重训 + 写提交文件 ----------
final = LGBMRegressor(objective='regression_l1', n_estimators=int(np.mean(best_iters) * 1.1),
                      learning_rate=0.02, num_leaves=31, colsample_bytree=0.8, subsample=0.8,
                      reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1)
final.fit(X, yy)
pred = final.predict(X_test)
if USE_LOG:
    pred = np.expm1(pred)
pred = np.maximum(pred, 0)

result = pd.DataFrame({ID_COL: test_ids, OUT_COL: pred})
result.to_csv('考生姓名_results.csv', index=False)
print('\n提交文件 考生姓名_results.csv 前 5 行:', flush=True)
print(result.head(), flush=True)
print('列名:', list(result.columns), '  ← 必须是 room_id 和 predicted_rrental_price_USD', flush=True)
