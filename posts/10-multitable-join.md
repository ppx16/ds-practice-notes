# 10 多表关联 + 欺诈识别

> 一句话总结：这题真正考的不是模型，是 join——标签表只有两列，特征全在另外七张表里，
> 合并顺序错一步，行数就悄悄变了。

| | |
|---|---|
| 题型 | 表格 · 二分类（多表关联 + 特征工程） |
| 数据 | 训练 16,312 行 × 2 列（只有交易编号 + 标签）；测试 4,079 行 × 1 列；输入字段要自己从 7 张来源表 join 出来（拼完是 15 列宽表） |
| 提交 | `202412_results.csv`，列：`TransactionID`, `Predicted_Results`（取值 `0` / `1`） |
| 评分 | 正类（欺诈 = 1）的 F1 |
| 代码 | [`code/10-multitable-join`](../code/10-multitable-join) · [`templates/10_表格二分类.py`](../templates/10_表格二分类.py) · [`templates/10_多表join_特征工程.py`](../templates/10_多表join_特征工程.py) |

## 一、题目在问什么

金融交易欺诈预测。但题目给训练/测试集的方式很特别：

- **训练集只有两列**：`TransactionID` 和 `FraudIndicator`；
- **测试集只有一列**：`TransactionID`。

也就是说，"输入字段"这道题**根本没有直接给**。你得自己从四个方向的来源数据里把特征拼出来：

| 来源方向 | 文件 | 拿到什么 |
|---|---|---|
| 交易数据 | `transaction_records.csv` | 金额、客户编号 |
| 交易数据 | `transaction_metadata.csv` | 时间戳、商家编号 |
| 顾客数据 | `customer_data.csv` | 姓名、性别、年龄、地址分区 |
| 顾客数据 | `account_activity.csv` | 账户余额、最近登录时间 |
| 商家数据 | `merchant_data.csv` | 商家名称、地点分区 |
| 商家数据 | `transaction_category_labels.csv` | 商品类别 |
| 异常数据 | `suspicious_activity.csv` | 该客户是否被标记可疑 |

题面本身就是这么说的：输入字段要自己靠特征工程建出来。
所以这题的分数几乎全压在**"表连得对不对"+"聚合特征造得好不好"**上。

## 二、数据长什么样

先把七张来源表看清（行数来自字段卡，非空都是满的）：

| 文件 | 行 × 列 | 字段（类型 / unique） |
|---|---|---|
| `transaction_records.csv` | 20,000 × 3 | `TransactionID`(obj, 20000) · `Amount`(float, 14918) · `CustomerID`(obj, **982**) |
| `transaction_metadata.csv` | 20,000 × 3 | `TransactionID`(obj) · `Timestamp`(obj, 19998) · `MerchantID`(obj, **20**) |
| `customer_data.csv` | 1,000 × 5 | `CustomerID`(obj) · `Name`(obj, 993) · `Sex`(obj, 2) · `Age`(int, 12) · `Address`(obj, 5) |
| `account_activity.csv` | 1,000 × 3 | `CustomerID`(obj) · `AccountBalance`(float, 999) · `LastLogin`(obj) |
| `merchant_data.csv` | **20** × 3 | `MerchantID`(obj) · `MerchantName`(obj, 20) · `Location`(obj, 5) |
| `transaction_category_labels.csv` | 20,000 × 2 | `TransactionID`(obj) · `Category`(obj, 7) |
| `suspicious_activity.csv` | 1,000 × 2 | `CustomerID`(obj) · `SuspiciousFlag`(int, 2) |
| `train_data.csv` | 16,312 × 2 | `TransactionID`(obj) · `FraudIndicator`(int, 2) |
| `test_data.csv` | 4,079 × 1 | `TransactionID`(obj) |

**从字段表就能读出四个关键结构**：

1. **粒度不一样。** 交易表是 20,000 行交易，客户表只有 1,000 个客户，
   商家表更是只有 20 个商家。`CustomerID` 只有 982 个唯一值 → 一个客户平均有 20 笔交易。
   **只要往交易级数据上 join 客户级/商家级字段，一定会一对多放大再收敛回来**，
   行数不变但每一行都会带上该客户/该商家的信息。
2. **`Address`、`Location`、`Sex`、`Category` 是低基数类别**（2~7 个取值），适合 One-Hot。
3. **`Name`、`MerchantName` 是纯标识**，一个客户一个名字，对预测没有任何用处，
   而且**属于个人信息，必须删掉**。
4. **时间有两列**：交易的 `Timestamp` 和客户最近登录的 `LastLogin`，
   两者相减是个很自然的"距上次登录多久"特征。

> 材料里其实还有两份同构的小数据（训练 800 / 测试 200，另外附带 `amount_data.csv`
> 和 `anomaly_scores.csv`），以及一份**带 `FraudIndicator` 的 `test_data.csv`**。
> 前者是别人放到网上的玩具数据，后者可以用来本地估分——但不带标签那份的列结构才和交卷时一致。

## 三、思路

**第一步，先把七张表连成一张宽表。** 主干选交易表，因为它粒度最细：

```text
transaction_records
  └─ + transaction_metadata (on TransactionID)   拿 Timestamp / MerchantID
  └─ + customer_data        (on CustomerID)      拿 Sex / Age / Address
  └─ + account_activity     (on CustomerID)      拿 AccountBalance / LastLogin
  └─ + merchant_data        (on MerchantID)      拿 Location
  └─ + transaction_category_labels (on TransactionID)  拿 Category
  └─ + suspicious_activity  (on CustomerID)      拿 SuspiciousFlag
```

**第二步，再把这张宽表按 `TransactionID` 挂到 train / test 上——顺序不能反。**
如果先拿 `train_data` 去掉别的表，就会因为各种一对多把 16,312 行撑大或者漏行。

**第三步，清洗。** 抢救被读成文本的数值列、去空格、空白当缺失、
数值列用中位数填、类别列用众数填，最后 LabelEncoder。

**第四步，特征工程。** 我这一版实际启用的是：

- 时间：`Timestamp` → datetime，可以取小时 / 星期 / 月份 / 是否夜间；
- 金额-账户交互：`Amount / AccountBalance`、是否透支；
- **交叉特征**：`SuspiciousFlag` × 金额 / × 是否大额 / × 占余额比 / × 是否深夜；
- 客户聚合：该客户的交易笔数、金额均值/最大/标准差、去过的商家数、类别数、活跃跨度天数；
- 商户聚合：该商家的交易笔数、金额均值、客户数；
- **OOF 目标编码**：给 `CustomerID`、`MerchantID` 这两个中高基数类别各算一列；
- 低基数 One-Hot：`Sex` / `Address` / `Location` / `Category`。

**第五步，建模。** LightGBM / XGBoost / RandomForest 各跑 5 折 OOF，
并且**给 LGBM/XGB 加早停**（早停验证集从每折训练集里再切 15%，验证折完全不参与），
三个模型按 OOF F1 挑，最后在 OOF 上搜门坎，输出 `0/1`。

## 四、关键代码

**① 六次 merge，主干选最细的那张表**：

```python
merged_data = transaction_records.merge(transaction_metadata, on="TransactionID", how="left")
merged_data = merged_data.merge(customer_data,             on="CustomerID",    how="left")
merged_data = merged_data.merge(account_activity,          on="CustomerID",    how="left")
merged_data = merged_data.merge(merchant_data,             on="MerchantID",    how="left")
merged_data = merged_data.merge(transaction_category_labels, on="TransactionID", how="left")
merged_data = merged_data.merge(suspicious_activity,       on="CustomerID",    how="left")

train = pd.read_csv("data/train_data.csv")
test  = pd.read_csv("data/test_data.csv")
train = train.merge(merged_data, on=ID_COL, how="left")     # 挂特征，不是反过来
test  = test.merge(merged_data, on=ID_COL, how="left")
```

`how="left"` 在这里不是可选项：用 `inner` 的话，只要某个 `TransactionID`
在来源表里缺一次，那笔交易就从训练集里凭空消失了，而你还看不出少在哪。

**② 客户级字段会按客户"传播"到该客户的所有交易上**：

```python
# suspicious_activity 是【客户】级的，join 到交易级后，
# 同一个客户的所有交易都会拿到同一个 SuspiciousFlag
full['susp_amount']        = full['SuspiciousFlag'] * full['Amount']
full['susp_large']         = full['SuspiciousFlag'] * full['is_large']
full['susp_amount_ratio']  = full['SuspiciousFlag'] * full['amount_ratio']
full['susp_over_balance']  = full['SuspiciousFlag'] * full['amount_over_balance']
```

这是这题最值钱的一段：单独看 `SuspiciousFlag` 只有 0/1 两个值的信息量，
但"可疑客户 × 大额"、"可疑客户 × 透支"这种交叉才表达出
"这个人本来就可疑，这笔还特别大"。`is_large` 用金额的 90 分位定义。

**③ OOF 目标编码（带平滑）**——`CustomerID` 有 982 个取值，直接 One-Hot 会把特征撑爆：

```python
def oof_target_encode(full, cat_col, target_col=TARGET, n_splits=5, smoothing=20, seed=42):
    ...
    def encode(ids, m_map, c_map):
        m = pd.Series(ids).map(c_map).fillna(0).values
        mean = pd.Series(ids).map(m_map).fillna(global_mean).values
        return (m * mean + smoothing * global_mean) / (m + smoothing)   # 平滑，防小样本类别乱跳

    te_enc = encode(te[cat_col].astype(str).values, st['mean'], st['count'])
    for tr_idx, va_idx in StratifiedKFold(n_splits=n_splits, ...).split(tr, y):
        fs = tr.iloc[tr_idx].groupby(cat_col)[target_col].agg(['mean', 'count'])
        tr_enc[va_idx] = encode(tr.iloc[va_idx][cat_col].astype(str).values, fs['mean'], fs['count'])
```

两个点：**训练集部分必须是 out-of-fold 算出来的**（否则每一行都用自己的标签编码自己，
目标编码会直接记住答案），**测试集部分用全训练集的统计量**。`smoothing=20` 让
样本数少的类别往全局均值收缩。

**④ 该丢的列一次丢干净**：

```python
full = full.drop(columns=['Timestamp', 'LastLogin', 'Name', 'MerchantName'])
```

时间列转成派生数字之后原始列就多余了；`Name` / `MerchantName` 是标识，
**姓名属于个人信息，不该进模型也不该出现在任何输出里**。

**⑤ 早停的验证集从训练折里再切，验证折不参与**：

```python
def oof_proba(m):
    for tr, va in skf.split(X, y):
        m2 = m.__class__(**m.get_params())
        if isinstance(m2, (LGBMClassifier, XGBClassifier)):
            X_tr, X_es, y_tr, y_es = train_test_split(
                X.iloc[tr], y[tr], test_size=0.15, stratify=y[tr], random_state=42)
            m2 = fit_es(m2, X_tr, y_tr, X_es, y_es)     # 早停用 X_es
        else:
            m2.fit(X.iloc[tr], y[tr])
        oof[va] = m2.predict_proba(X.iloc[va])[:, 1]
        test_p += m2.predict_proba(X_test)[:, 1] / 5
```

**早停的验证集绝对不能是 `va`**，否则最优轮数就是"看着答案挑的"，
OOF 分数会虚高，而这份虚高正好会误导后面的门坎搜索。

## 五、踩的坑

**坑 1：`templates/10_表格二分类.py` 里有一段"merge 完又被覆盖"的代码。**
那次六次 merge 之后紧接着：

```python
merged_data = pd.read_csv("merged_data.csv")      # ← 前面六次 merge 的结果被整段覆盖
```

而且那一版的 merge 用的是 `how="inner"`。这意味着**那份代码真正用的数据其实是预先存好的
`merged_data.csv`**，前面那三十来行 merge 只是写给自己看的"流程注释"。
这种代码最危险的地方是它**能跑通、也不报错**，只是用的数据和你以为的不是同一份。
[`templates/10_多表join_特征工程.py`](../templates/10_多表join_特征工程.py) 里才改成
"merge 出来直接用"的干净写法——它的文件头也写明了那一版是在前一版基础上继续加东西。

**坑 2：join 的方向。** 正确顺序是"先拼宽表，再挂到 train/test"。
反过来（先拿 train 去掉来源表）在有一对多关系时行数会变，
而 pandas 只会给你一个变长了的 DataFrame，不会给你任何警告。
后来我把"挂完之后检查 `len(train)` 有没有变"加进了习惯动作。

**坑 3：材料里那份小数据集是别人用 `np.random` 造的。**
`code/10-multitable-join/code/Dataset/` 里有一份外部脚本来生成数据，
它的欺诈标签是 `np.random.choice([0, 1], 1000, p=[0.95, 0.05])`、
可疑标记是 `p=[0.98, 0.02]`——**标签是随机投的**。
也就是说那份数据里根本不存在可学的规律，在那上面调模型是纯浪费时间。
只有 `TRANS...` / `CUST...` 编号的那一套（16,312 训练 / 4,079 测试）才是练习用的数据。
**看到"商家的名字是随机单词组合、只有 20 个商家、Age 只有 12 个取值"这种特征，
就该先怀疑数据是不是合成的**，而不是埋头调参。

**坑 4：第三方的参考 notebook 不是我的代码，别当成自己的结论。**
`code/10-multitable-join/code/Dataset/` 和 `.../Dataset2/` 下的
`Solutions.ipynb`、`fraud-detection-frostboost.ipynb` 是**从外面下载的分享 notebook**
（连数据集自带的 `Readme.md`、生成脚本都在一起），属**外部参考资料**。
它们的内容是：先做同样的多表合并，然后重点比较 **SMOTE / FROST 过采样**，
再配 `GridSearchCV` 横向跑 DecisionTree / XGBoost / Bagging / RandomForest / LogisticRegression / 神经网络。

它们自己的输出里有一组数字很值得记：
**同一份数据、同一个 RandomForest，训练集 F1 能到 0.88~0.94，而留出的测试集 F1 只有 0.47~0.57。**
这个落差比任何结论都说明问题——**过采样只改变训练集的类比例，不产生新信号**。
所以我最后没走 SMOTE 那条路，而是用 `scale_pos_weight` / `class_weight` 把权重提到模型里，
把力气花在 join 出来的聚合特征上。

**坑 5：提交取值是数字 `0/1`，不是字符串。**
这题的 `Predicted_Results` 要写 `0` / `1`；
我在另一道同类练习（客户流失）上交的是 `Yes` / `No`，电商那题又要 `TRUE` / `FALSE`。
同一套流程，三种写法，**每次都要照题目要求逐字确认**。

## 六、小结

- 多表题的心法就一句：**主干选粒度最细的那张表，一路 `left join` 补字段，
  join 完再挂标签**。顺序对了，后面才有意义。
- 客户级/商家级字段 join 到交易级之后**是被"传播"的**，
  所以要靠交叉特征（可疑 × 金额 × 时段）把这种传播变成可用的信号。
- **中高基数类别用带平滑的 OOF 目标编码，低基数类别直接 One-Hot**，
  这条分工后来也被写进了特征工程片段库（片段③和片段⑦）。
- 这题教会我一个以前不会做的动作：**先看数据的"长相"判断它是不是合成的**。
  `Age` 只有 12 个取值、商家只有 20 个、标签 5% 随机——这些迹象说明
  再怎么调参也不会更好，早点发现能省下大把时间。
- 那两份外部的参考 notebook 我只当作"看别人怎么处理这类题"的读物，
  **里面的代码和分数都不是我的结果**，仓库里留着它们只是为了保留来源痕迹。
