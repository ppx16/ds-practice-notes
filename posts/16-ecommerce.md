# 16 电商用户行为预测

> 一句话总结：这一题的 AUC 只有 0.63——我第一次遇到"特征工程做完还不如不做"的数据集，
> 也第一次把整条流水线写成了一份能换数据直接跑的骨架。

| | |
|---|---|
| 题型 | 表格 · 二分类 |
| 数据 | 训练 280,000 行 × 20 列；测试 70,000 行 × 19 列（19 = 20 去掉目标列） |
| 提交 | `考生姓名_results.csv`，列：`ID`, `Predicted_Results`，取值只能是 `TRUE` / `FALSE` |
| 评分 | 正类（`Revenue = TRUE`）的 F1 |
| 代码 | [`code/16-ecommerce`](../code/16-ecommerce) · [`templates/16_表格二分类.py`](../templates/16_表格二分类.py) · [`templates/16_结构化实战模板.py`](../templates/16_结构化实战模板.py) |

**先说一句实话**：这题的代码**几乎全在 `templates/` 下面**，不在 `code/` 里。
`code/16-ecommerce/` 只有一个 notebook（`test.ipynb`），是当时的建模过程记录；
真正沉淀下来的东西是 `templates/16_*.py` 那一整套。所以这篇复盘的重点会放在模板上。

## 一、题目在问什么

给电商平台的一次会话记录：这次会话里用户看了几类页面、各停了多久、跳出率/离开率、
页面价值、流量来源、操作系统、浏览器、地区、是不是周末、是不是节庆附近……
判断**这次会话最后会不会成交**。

题面交代的三个难点：

1. 有缺失，而且三大主力特征（页面停留时长、跳出率、页面价值）**缺得最多**；
2. 样本不平衡——大多数人只是逛逛，成交的是少数；
3. 类别变量维度不低（月、操作系统、浏览器、地区、流量来源、访客类型、星座），
   还混了**明显不相关的噪声列**。

## 二、数据长什么样

训练 20 列 = `ID` + 18 个特征 + 目标 `Revenue`。

| 列名 | 类型 | 非空/20000 | unique | 说明 |
|---|---|---|---|---|
| `ID` | int64 | 20000 | 20000 | 会话流水号，全唯一 |
| `Administrative` | int64 | 20000 | 11 | 浏览行政类页面次数（计数） |
| `Admin_Duration` | float64 | 13982 | 13982 | 行政页停留总时长，**约 30% 缺失** |
| `Informational` | int64 | 20000 | 7 | 信息类页面次数（计数） |
| `Info_Duration` | float64 | 20000 | 20000 | 信息页停留总时长 |
| `ProductRelated` | int64 | 20000 | 43 | 产品页次数（计数，最主力的一列） |
| `Product_Duration` | float64 | 20000 | 20000 | 产品页停留总时长（**信号最强**） |
| `BounceRates` | float64 | 14074 | 14074 | 跳出率 0~1，**约 30% 缺失** |
| `ExitRates` | float64 | 20000 | 20000 | 离开率 0~1 |
| `PageValues` | float64 | 14057 | 14057 | 页面价值均值，**约 30% 缺失，也是最强特征之一** |
| `SpecialDay` | float64 | 20000 | 4 | 距节庆程度，只有 4 个取值 |
| `Month` | object | 20000 | 10 | 会话月份，Jan~Dec 缺两个 |
| `OperatingSystems` | int64 | 20000 | 7 | 操作系统代码（**是类别，不是数值**） |
| `Browser` | int64 | 20000 | 12 | 浏览器代码（同上） |
| `Region` | int64 | 20000 | 8 | 地区代码（同上） |
| `TrafficType` | int64 | 20000 | 19 | 流量来源代码（同上） |
| `VisitorType` | object | 14076 | 3 | 新访客 / 回头客 / 其他，**三成缺失** |
| `Weekend` | bool | 20000 | 2 | 是否周末 |
| `User_Horoscope` | object | 20000 | 6 | 用户星座——**纯噪声** |
| `Revenue` | bool | 20000 | 2 | 目标，`True` / `False` |

三个从字段表就能读出结论的点：

- **缺得最多的三列恰好是最重要的三列**，所以缺失不能简单丢掉，得当成信号处理；
- `OperatingSystems` / `Browser` / `Region` / `TrafficType` 的 dtype 是 int64，
  但它们是**编码后的类别**，当成连续数值喂进树模型是错的；
- `User_Horoscope` 这一列看着像有信息，实际上是在钓人（下面第四节说它的实测结果）。

> 注：字段卡的 unique/非空列是按 2 万行抽样统计的，规模以 notebook 打印的
> `合并后 shape: (350000, 20)` 为准（28 万 + 7 万）。

## 三、思路

notebook（`code/16-ecommerce/test.ipynb`）的路线是：

1. **合并训练/测试再一起清洗**，最后按目标列是否为 NaN 拆回去——这样两边的编码码表一定一致。
2. **缺失处理分两类**：
   - 数值列（`Admin_Duration` / `BounceRates` / `PageValues`）→ 先加一列 `xxx_isna` 记录
     "这里是缺失"，再用**训练集的中位数**填（故意不用全表中位数，防止测试集信息漏进来）；
   - 类别列 `VisitorType` → 把缺失当成一个新类别 `Unknown`。
3. **缩尾**：对 6 个长尾连续列按训练集的 0.5% / 99.5% 分位 clip。
4. **特征工程**：总时长、总页面数、平均每页停留、各类页面占比、各类时长占比、
   跳出/离开组合、`PageValues` 的 log、节庆高低、月份顺序编码。
5. **类别处理**：`Month` / `OperatingSystems` / `Browser` / `Region` / `TrafficType` /
   `VisitorType` 全转成 pandas `category`，**交给 LightGBM 原生处理**，不自己做 One-Hot。
6. **调参**：Optuna 贝叶斯搜索 30 组，目标函数直接就是"OOF 上最优门坎下的 F1"。
7. **5 折 OOF 出概率 → 在 OOF 上搜门坎 → 全量重训 → 测试集概率取
   "5 折平均"和"全量模型"各 0.5 的融合**。

## 四、关键代码

### 4.1 缺失列：指示特征 + 只用训练集中位数

```python
num_cols_with_na = ['Admin_Duration', 'BounceRates', 'PageValues']
for c in num_cols_with_na:
    full[c + '_isna'] = full[c].isna().astype(int)        # 缺失本身可能是信号
    med = full.loc[full['Revenue'].notna(), c].median()    # 只用训练集算中位数
    full[c] = full[c].fillna(med)

full['VisitorType'] = full['VisitorType'].cat.add_categories(['Unknown']).fillna('Unknown')
```

`full['Revenue'].notna()` 这个掩码就是"取训练集"——用全表算中位数是很多人（包括当时的我）
第一反应会写错的地方。

### 4.2 类别列直接交给 LightGBM，别自己做 One-Hot

```python
for c in ['OperatingSystems', 'Browser', 'Region', 'TrafficType']:
    full[c] = full[c].astype('category')

cat_cols = [c for c in feature_cols if str(X[c].dtype) == 'category']
```

这四个列名的 dtype 是 int64，看着像数字，其实是类别码。一个改动带来两个效果：
不用手工 One-Hot，也不会让模型误以为"浏览器 12 比浏览器 3 大"。

### 4.3 Optuna 的目标函数直接对齐评分

```python
def objective(trial):
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 0.3, log=True),
        'num_leaves':    trial.suggest_int('num_leaves', 15, 255),
        'max_depth':     trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'reg_alpha':     trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
    }
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        sw = np.where(y_tr == 1, (y_tr == 0).sum() / (y_tr == 1).sum(), 1.0)
        m = lgb.LGBMClassifier(n_estimators=1000, **params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False)], sample_weight=sw)
        oof[va_idx] = m.predict_proba(X_va)[:, 1]
    # 在 OOF 上搜索最优门坎, 返回其 F1 (对齐评分)
    best_f1 = 0.0
    for thr in np.arange(0.05, 0.75, 0.01):
        f1 = f1_score(y, (oof >= thr).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    return best_f1
```

**每一步 trial 内部都跑了完整 5 折，并且在 OOF 上搜门坎才返回 F1。**
这么写比"最小化 logloss"贵得多，但它优化的就是评分指标本身。
记录里这次搜索的最优 OOF F1 是 `0.2930`。

### 4.4 门坎优化 + 两种概率融合

```python
best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.05, 0.95, 0.005):
    f1 = f1_score(y, (oof_proba >= thr).astype(int))
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

# 测试集概率：5 折平均与全量模型各占一半
test_proba_final = 0.5 * test_proba + 0.5 * final_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba_final >= best_thr).astype(int)
```

记录里 `best_thr = 0.515`，OOF F1 `0.2919`，而默认 0.5 是 `0.2913`——
**这题的门坎优化几乎白干**，因为概率分布本来就挤在一起。

### 4.5 16 家族：把这一整套固化成可以换数据直接跑的骨架

`templates/16_*.py` 是这条流水线最终的样子，一共六个文件，分工很清楚：

| 文件 | 定位 |
|---|---|
| [`16_表格二分类.py`](../templates/16_表格二分类.py) | 这题的"专门版"：月份顺序编码 + 电商专属衍生特征 + 反编码成 `True`/`False` |
| [`16_速通版.py`](../templates/16_速通版.py) | 30 分钟保底：读数据 → 补缺失 → 编码 → LGBM 3 折 → 搜门坎 → 写文件 |
| [`16_特征工程片段库.py`](../templates/16_特征工程片段库.py) | 11 个可拷贝片段 + 一张"看列选招式"的速查表 |
| [`16_自动检测特征工程.py`](../templates/16_自动检测特征工程.py) | 喂它一个 CSV，逐列判型并**生成列名已填好的片段代码** |
| [`16_特征工程已填好.py`](../templates/16_特征工程已填好.py) | 上一步在本数据集上跑出来的产物 |
| [`16_结构化实战模板.py`](../templates/16_结构化实战模板.py) | **集大成者**，把前面所有练习的教训合并成一个可配置骨架（1153 行） |

速通版的存在本身就是一条教训：考场上最先要做的事是"先有一个能交的结果"。
它只保留最短路径，`16_速通版.py` 的注释里写着这份东西的用途是"确保有结果能交"，
之后再从主模板里按需抄功能段加强。

片段库的用法不是"全都贴上"，而是先看第 0 节的速查表，判断一列属于
"计数 / 金额 / 时长 / 比率 / 等级 / 有序类别 / 无顺序类别 / 日期 / 文本 / 纯序号"里的哪一类，
再决定用哪个片段。它的性价比排序是：**比率差值 > 行统计 > OOF 目标编码 > 有序映射 > 频次编码**。
`自动检测特征工程.py` 把这一步自动化了：它按列名关键词 + 数据特征给每列判型，
再用互信息挑出最值得做比率/差值的字段对。它在电商数据上生成的产物就在
`16_特征工程已填好.py` 里——排名第一的那一对是 `ProductRelated` × `Product_Duration`。

主模板 `16_结构化实战模板.py` 值得单独说的几处：

```python
# ① 正类怎么定 + 提交里该写什么字样，全部自动推导
def resolve_sub_format(pos_raw, neg_raw, mode=None):
    p, n = fmt_label(pos_raw), fmt_label(neg_raw)
    if {p.lower(), n.lower()} == {'0', '1'}:
        return p, n, "自动判定：原始目标是数字 0/1 -> 写 0/1"
    if {p.lower(), n.lower()} == {'true', 'false'}:
        return 'TRUE', 'FALSE', "自动判定：原始目标是布尔 -> 写 TRUE/FALSE"
```

这是被"提交取值大小写"反复坑过之后的产物：同一套流水线，
在练习 15 要交 `Yes`/`No`、在练习 10 要交 `0`/`1`、在这一题要交 `TRUE`/`FALSE`。
模板改成从原始目标值的字面去推导，而不是写死一种。

```python
# ② ID 列强制按字符串读（否则前导零会被 pandas 吃掉，ID 全对不上 = 0 分）
kw = {'dtype': {id_col: str}} if id_col else {}
```

```python
# ③ 门坎搜索：排序后一次扫描 O(n log n)，并且只在"并列概率组的最后一个位置"取候选
order = np.argsort(-p, kind='mergesort')
tp = np.cumsum(y_true[order])
f1 = np.divide(2.0 * tp, 2 * tp + (k - tp) + (tp[-1] - tp), ...)
last_of_group = np.where(np.r_[ps[:-1] != ps[1:], True])[0]
```

这一段的注释里记着两个真实教训：逐门坎重算 F1 是 O(n²)，28 万行时"看起来像卡死"；
以及**并列概率会让扫描出来的 F1 和用 `>=` 复算的 F1 对不上**
（记录里出现过 F1 从 0.2388 变成 0.0000 的情况），所以候选门坎只能落在完整并列组的边界上，
并且写完还要用 `f1_score` 复核一次。

```python
# ④ 预测全是同一类 = 0 分，所以宁可自动补救也不让它崩
if _raw_check[SUB_COL].nunique() < 2:
    _k = float(np.clip(y.mean(), 0.02, 0.98))
    _cut = np.quantile(final_test, 1 - _k)     # 改成"按概率取前 K% 判为正类"
    sub = write_submission(final_test, _cut, SUB_PATH)
```

```python
# ⑤ 最后一道格式自检：列名、行数、缺失、取值是否只有那两种、是不是全同一类
assert len(raw.columns) == 2 and raw.columns[0] == ID_COL, '列名不对！'
assert len(raw) == len(X_test), '行数对不上！'
assert set(vals) <= {SUB_POS, SUB_NEG}, '取值不对（大小写要照要求）！'
assert raw[SUB_COL].nunique() >= 2, '全部预测同一个值 = 0 分！'
```

主模板还有一个设计我很喜欢：**加分实验只在确实更好时才被采用**。
比如不平衡策略那条，会额外跑一版"LGBM 不加权 + 门坎优化"，只有 OOF F1 更高才替换当前方案。
其中"特征工程对比"那段特意修过一次 bug：基线必须**包含类别编码列**，
否则对比出来的差值里混进了编码的功劳，数字会虚高。

## 五、踩的坑

**坑 1：这题的特征工程做完可能更差。** 主模板的注释里留了一条实测记录：
"我在这份电商数据上实测过：基线 0.2716 -> 加特征 0.2619"。
主模板因此专门加了一条规矩——粘完片段要用同一份切分做 A/B 对比，**只有提升才保留**，
否则把片段删掉、用干净数据交卷。这跟我以前的直觉完全相反。
原因也不难理解：AUC 只有 0.63 的数据集里，衍生特征大多在放大噪声。

**坑 2：信号是真的弱。** notebook 的 5 折记录里，AUC 全在 0.637 上下
（OOF AUC = 0.6357），最好的 F1 大约 0.29。用户星座这一列，notebook 的注释写得很直接：
各星座购买率几乎一致（约 13.5%），所以判为干扰特征剔除。**特征重要性排名靠前的
`TrafficType`、`Browser` 我更倾向于当成过拟合的迹象，而不是"发现了新知识"**。

**坑 3：概率被整体压低，全同一类的风险是真的存在。** 正类只占 13.5%，
如果门坎没调好，很容易出现"预测全是 FALSE"。一旦如此，文件格式再对也是 0 分。
这就是主模板 4.4 那段自动补救的来由。

**坑 4：提交列名写成了小写。** notebook 最后写出去的是：

```python
submission = pd.DataFrame({'id': test_df['ID'].values,
                           'Predicted_Results': np.where(test_pred == 1, 'TRUE', 'FALSE')})
```

`Predicted_Results` 的取值写对了，但 ID 列名写成了 `id`，而题目要的是 `ID`。
同一份数据、同一套流程，第 10 题要 `TransactionID`、第 15 题要 `id`、这一题要 `ID`——
**列名必须逐个题目照抄，不能靠记忆**。

**坑 5：`Category` 的坑我在别的表上也踩过。** 把"编码型 int64"当数值列用，
树模型能忍，但线性和神经网络就忍不了。主模板里现在有一句自动纠偏：
`for c in ['id','ID','Id','customerID','用户ID']` 按候选名找 ID 列，
以及把疑似日期/类别的列单独拦出来处理。

## 六、小结

- 这题最大的收获不是分数，而是**"特征工程不总是加分"这个反直觉事实**。
  后来我在骨架里加了强制 A/B 对比，就是为了不再凭感觉往模型里塞列。
- **AUC 0.63 的数据上，能做的只有两件事**：把预处理做干净（缺失指示、缩尾、类别当类别），
  以及把门坎调到评分指标最优。调参换来的提升远小于这两个。
- 这一题的代码主体在 `templates/16_*.py`，`code/16-ecommerce/` 只有一份建模过程 notebook。
  **[`templates/16_结构化实战模板.py`](../templates/16_结构化实战模板.py) 是这套模板的终点**：
  配置区只留四个必改项（数据路径、目标列、ID 列、正类），其余全是这几年被坑出来的自动化和自检——
  ID 按字符串读、提交字样自动推导、日期列自动转数值、门坎 O(n log n) 扫描、
  RF 树数按数据量缩放、线程数自动取半、预测全同一类自动补救、写完还有 assert 自检。
  它同时也是一份性能笔记：文件头记着"三个速度修复让总耗时从 223s 降到 29s"、
  以及 GPU 在多少行数据上才开始划算。
- 以后再遇到"看起来字段很多但 AUC 上不去"的表，我的顺序会是：
  先跑速通版拿到保底提交 → 再看字段分型 → 只贴性价比最高的 2 个片段 →
  同一份切分做 A/B → 门坎对齐评分指标。**而不是一上来就把所有特征都造一遍。**
