# 15 客户流失预测

> 一句话总结：这题让我记住两件事——「被读成字符串的金额列」一定要抢救，以及在不平衡数据上看准确率等于自欺欺人。

| | |
|---|---|
| 题型 | 表格 · 二分类 |
| 数据 | 训练 389,712 行 × 22 列；测试 97,428 行 × 21 列（21 = 22 去掉目标列） |
| 提交 | `考生姓名_results.csv`，列：`id`, `Predicted_Results`，取值只能是 `Yes` / `No` |
| 评分 | 正类（`Churn = Yes`）的 F1 |
| 代码 | [`code/15-churn`](../code/15-churn) · [`templates/15_表格二分类.py`](../templates/15_表格二分类.py) |

## 一、题目在问什么

给一份电信客户的资料表：性别、是否老人、有没有家属、订了哪些服务、合约类型（月付 / 一年 / 两年）、
付款方式、月费、累计消费……要求预测**这个客户会不会流失**。

题面自己交代了这份数据是"故意做脏"的：

- 有缺失（而且缺失不是随机的，很多字段只有 8 成左右的非空）；
- 有异常值，最典型的是**空白字符串**当缺失用；
- 有非数值型态——金额列被写成了文本；
- **有干扰特征**：`FavoriteColor`（客户最喜欢的颜色）、`LuckyNumber`（幸运数字）这两列跟业务毫无关系，
  摆明了是来钓"无脑把所有列丢进模型"的人。

评分只看 **流失那一类的 F1**，所以"全部预测不流失"拿不到任何分——这一点决定了后面所有设计。

> 行数说明：我本地抽的字段卡是 **2 万行抽样**，notebook 跑的是全量 389,712 / 97,428。
> 下面字段表的 unique 数来自抽样卡，规模以 notebook 打印为准。

## 二、数据长什么样

训练 22 列 = 1 个 ID + 20 个特征 + 1 个目标 `Churn`。

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `id` | int64 | 20000 | 客户 ID，全唯一，提交时要原样带回去 |
| `gender` | object | 3 | 性别。**unique=3 就说明有问题**——除了 Female/Male 还混进了别的取值 |
| `SeniorCitizen` | object | 3 | 是否老年人，原始值是 0/1 却被存成文本 |
| `Partner` | object | 3 | 是否有伙伴 |
| `Dependents` | object | 3 | 是否有家属 |
| `FavoriteColor` | object | 5 | 最喜欢的颜色——**纯噪声** |
| `LuckyNumber` | float64 | 1000 | 幸运数字——**纯噪声** |
| `PhoneService` | object | 3 | 是否订购电话服务 |
| `MultipleLines` | object | 4 | 是否多条线路 |
| `InternetService` | object | 4 | 上网方式（DSL / Fiber optic / No） |
| `OnlineSecurity` | object | 4 | 网络安全服务 |
| `OnlineBackup` | object | 4 | 在线备份 |
| `DeviceProtection` | object | 4 | 设备保护 |
| `TechSupport` | object | 4 | 技术支持 |
| `StreamingTV` | object | 4 | 流媒体电视 |
| `StreamingMovies` | object | 4 | 流媒体电影 |
| `Contract` | object | 4 | 合约期限：月付 / 一年 / 两年（**有序**） |
| `PaperlessBilling` | object | 3 | 无纸化账单 |
| `PaymentMethod` | object | 5 | 付款方式，含 "automatic" 字样 |
| `MonthlyCharges` | float64 | 1513 | 月费 |
| `TotalCharges` | object | 7165 | 累计消费。**本该是数值，被读成了 object** |
| `Churn` | object | 2 | 目标，取值 `Yes` / `No` |

几个直接能读出信息的点：

1. 22 列里 16 列是 object，其中真正"类别型"的只有一部分，`TotalCharges`、`LuckyNumber`、`MonthlyCharges`
   是被脏数据坑成文本的数值列。
2. 大量 `unique=3` 的列，正常应该是 2（是/否），多出来的那一类就是空白串。
3. **没有"使用月数"字段**。但 `TotalCharges / MonthlyCharges` ≈ 在网月数——这是这题唯一一个
   真需要业务理解才能想到的衍生特征。

## 三、思路

1. **先清洗类型**：把该是数值的三列（`LuckyNumber` / `MonthlyCharges` / `TotalCharges`）强制转数值，
   转不动的变 NaN，再用**中位数**填；类别列统一填 `'Unknown'`。
2. **丢噪声列**：`FavoriteColor`、`LuckyNumber` 直接 drop。
3. **造 `tenure`**：用 `TotalCharges / MonthlyCharges` 反推在网月数，注意分母为 0 的保护。
4. **编码**：类别列用 `LabelEncoder`，而且训练集和测试集的取值**合并起来一起 fit**，
   否则测试集出现训练集没见过的类别时会直接报错。
5. **衍生特征**：平均月消费、月费与平均月费的差、总消费/月费比、是否新客（tenure≤6）、
   是否长期客户（tenure≥48）、附加服务个数、tenure 分桶。
6. **多模型对比**：LightGBM / XGBoost / RandomForest / GradientBoosting 四个都跑，
   在 8:2 分层留出集上看 AUC / F1。
7. **阈值搜索**：用 PR 曲线找让 F1 最大的阈值，而不是死守 0.5。
8. **全量重训 + 出提交**。

不平衡处理贯穿全程：notebook 里算出 `scale_pos_weight = 17.2`（负样本是正样本的 17 倍），
树模型都用这个权重。

## 四、关键代码

**① 抢救"被读成文本的数值列"**——这题最重要的 5 行：

```python
def clean_numdata(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')   # 空白串 -> NaN
    return df

train_df = clean_numdata(train_df, cols=['LuckyNumber', 'MonthlyCharges', 'TotalCharges'])
train_df = fill_numdata(train_df, cols=['LuckyNumber', 'MonthlyCharges', 'TotalCharges'])
```

`errors='coerce'` 是关键：转不动的值不会抛异常，而是变成 `NaN`，交给下一步中位数填。
如果不做这一步，`TotalCharges` 会被当成第 17 个类别列做 LabelEncoder，
**累计消费的数值信息就整列丢掉了**。

**② 用两个字段反推出题目没给的"使用月数"**：

```python
def MakeNewCol(data):
    # 构造新特征：使用时长 ≈ TotalCharges / MonthlyCharges
    data['tenure'] = 0.0
    mask = data['MonthlyCharges'] > 0                 # 分母为 0 的保护
    data.loc[mask, 'tenure'] = data.loc[mask, 'TotalCharges'] / data.loc[mask, 'MonthlyCharges']
    return data

train_df = MakeNewCol(train_df)
test_df  = MakeNewCol(test_df)
```

先用 `mask` 把月费为 0 的行排除掉再加，比 `fillna` 之后再除要稳，也不用担心 `inf`。

**③ 编码时把测试集的取值并进来**：

```python
le = LabelEncoder()
all_vals = set(X_train[col].dropna().unique()) | set(X_test[col].dropna().unique())
le.fit(list(all_vals))
X_train[col] = le.transform(X_train[col].astype(str))
X_test[col]  = le.transform(X_test[col].astype(str))
```

因为在 2.4 步里类别缺失已经统一填成 `'Unknown'`，这里其实不会遇到没见过的值；
但这个写法是我的固定习惯——**它把"训练/测试码表不一致"这类事故从根上掐掉了**。

**④ 用 PR 曲线找阈值，不用 0.5**：

```python
y_prob = model.predict_proba(X)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh_pr = thresholds[best_idx]

y_test = model.predict_proba(X_test)[:, 1]
y_test = (y_test >= best_thresh_pr).astype(int)
```

正类只占 5.5%，概率被整体压低，0.5 这个门槛会把绝大多数流失客户判成"不流失"。
notebook 里的记录很直观：LightGBM 在默认 0.5 下 F1 是 `0.33694`，换成 PR 曲线选的阈值后变成 `0.40275`。

**⑤ 不平衡权重**：

```python
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count    # 记录里打印出来是 17.2023
```

## 五、踩的坑

**坑 1：准确率会骗人。** notebook 里 GradientBoosting 的记录是
`accuracy_score_0: 0.94691`，但同一行的 `F1_0: 0.18575`。
因为不流失的客户占 94.5%，全猜"不流失"就有 94.5% 的准确率——
准确率高只说明我会背多数类，不代表模型学到了任何东西。这题只该看 F1 和 AUC。

**坑 2：`FavoriteColor`、`LuckyNumber` 必须删。**
"最喜欢的颜色"和"幸运数字"对"会不会退订"没有任何可解释的作用，
正常业务里也不会拿这两项做客户挽留。notebook 里是直接
`train_df.drop(['FavoriteColor','LuckyNumber'], axis=1)` 处理掉的。
树模型不嫌特征多，塞进去它照样给你分裂，白白增加过拟合面。

**坑 3：提交格式和 notebook 的输出对不上。** 这是事后复查才发现的：
题目要求的是两列 `id` / `Predicted_Results`（`Yes` / `No`），
而 notebook 最后写出去的其实是 `id` / `Churn`（0/1）：

```python
submission_opt = pd.DataFrame({'id': test_ids, 'Churn': y_test})
```

分类效果另说，**光是列名和取值对不上，这份文件就是 0 分**。
后来我把这部分统一收进了 [`templates/15_表格二分类.py`](../templates/15_表格二分类.py)：
模板里把"预测 0/1 → 反编码回题目要求的字样"单独放在一个标注了"改这里"的区块，
就是被这次坑出来的。这个坑在 4 个练习里的表现还不一样：本练习要 `Yes`/`No`，
多表 join 那题要 `0`/`1`，电商那题要 `TRUE`/`FALSE`——**同一个动作，三种写法**。

**坑 4：optuna 调参那一段是空的。** notebook 里留了完整的搜索空间定义，
但整段被注释掉了，最后也只写了一句"有时间的话……"。所以：
**这题的调参我没有真正跑过，模型用的是手写默认参数**，不能拿它当"调过参"的结论。

**坑 5：堆叠融合那一段有个我自己都不放心的写法。** 手动加权平均之后，
又用一层 LogisticRegression 做元模型，最后按 OOF AUC 挑一个方案。
但代码里 `meta_model.fit(test_preds, y)` 用的是**测试集预测**去训元模型，
而 `final_predictions_prob = meta_test` 又是同一条路径算出来的——这段逻辑前后不一致，
我在注释里也标了"更合理的做法是用 OOF"。这段当时没跑出干净结论，先记在这。

## 六、小结

- 这题的技术含量集中在**清洗**，不在建模：3 个脏数值列 + 一堆空白串 + 2 个噪声列，
  清完之后模型选谁其实差别不大（4 个模型 AUC 都在 0.894~0.899 之间）。
- **评分指标决定一切设计**：因为只看正类 F1，"阈值优化"比"换模型"值钱得多。
- 提交文件的**列名和取值大小写**是硬门槛，必须在代码最后一段加自检。
  这套自检我后来写进了模板和 16 那套骨架里。
- 事后我把这一整套（抢救数值列 → 填缺失 → 编码 → 衍生 → 多模型 OOF → 搜阈值 → 写文件）
  固化成 [`templates/15_表格二分类.py`](../templates/15_表格二分类.py)，
  再看这题时只需改数据路径、目标列名、ID 列名和反编码字典四处。
