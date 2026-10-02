# 04 住宿日租价格预测

> 25 个字段里有 4 类完全不同的东西（数值 / 分类 / t-f 布尔 / 设施列表），
> 混着当一坨喂模型是错的；另外提交列名带个双 r 的拼写，写错就是 0 分。

| | |
|---|---|
| 题型 | 回归 |
| 数据 | 训练 4,667 行 × 27 列，测试 1,167 行 × 27 列（练习这份带目标列，能直接算分） |
| 提交 | `<姓名>_results.csv`，列：`room_id`, `predicted_rrental_price_USD` |
| 评分 | RMSE |
| 代码 | [`code/04-rental-price`](../code/04-rental-price) · [`templates/04_回归_改进版.py`](../templates/04_回归_改进版.py) |

## 一、题目在问什么

给一批共享住宿房源的信息，预测它的**日租美金价格**。
输入按题目自己的说法分成五组：租赁特征（可住几天、取消政策等）、
房源特征（能住几人、几间卧室、有什么设施）、房主特征（房主挂了几套房、是不是超赞房东）、
位置特征（经纬度）、声誉特征（各类评分）。

目标列是 `rental_price_USD`，连续值，评分按 RMSE 排名。

提交两列：`room_id` 和 `predicted_rrental_price_USD` ——
**注意这里是双 r 的 `rrental`**，是题目原文里的拼写。这一条我在模板文件里专门标了星号：
列名写错就是格式错误，跟模型好坏无关，直接归零。

配套还有一张 `description.xlsx`，25 行字段说明表（`欄位` / `欄位特性` / `說明` / `範例`），
它不参与建模，只是告诉我每个字段属于哪一组特征、中文意思是什么。

## 二、数据长什么样

`training.xlsx`（4667 行 × 27 列，其中 25 个输入字段 + `room_id` + 目标）：

| 列名 | 类型 | 非空 | unique | 说明 |
|---|---|---|---|---|
| `room_id` | int64 | 4667 | 4667 | 房间序号（主键，建模时当索引，不进特征） |
| `accommodates` | int64 | 4667 | 16 | 可容纳人数 |
| `amenities` | object | 4667 | 3666 | 设施列表，形如 `{TV,Internet,Wireless Internet,...}` |
| `availability_30` | int64 | 4667 | 31 | 未来 30 天可订天数 |
| `bathrooms` | float64 | 4632 | 15 | 浴室数（有缺失） |
| `bed_type` | object | 4667 | 5 | 床型，如 Real Bed / Futon |
| `bedrooms` | float64 | 4662 | 10 | 卧室数（有缺失） |
| `beds` | float64 | 4649 | 16 | 床位数（有缺失） |
| `calculated_host_listings_count` | int64 | 4667 | 14 | 房主名下房源数（平台算的） |
| `cancellation_policy` | object | 4667 | 5 | 取消政策 |
| `guests_included` | int64 | 4667 | 16 | 房价包含的客人数 |
| `has_availability` | object | 4667 | 1 | 是否可订，全是 `t` —— **零信息列** |
| `host_is_superhost` | object | 4654 | 2 | 是否超赞房东（`t`/`f`） |
| `host_listings_count` | float64 | 4654 | 24 | 房主自填的房源数（有缺失） |
| `instant_bookable` | object | 4667 | 2 | 是否可即时预订（`t`/`f`） |
| `latitude(North)` | float64 | 4667 | 4666 | 纬度 |
| `longitude(East)` | float64 | 4667 | 4660 | 经度 |
| `maximum_nights` | int64 | 4667 | 78 | 最长可住夜数 |
| `number_of_reviews` | int64 | 4667 | 152 | 评论数 |
| `property_type` | object | 4667 | 16 | 房源类型，如 House / Camper/RV |
| `review_scores_checkin` | float64 | 3027 | 7 | 入住体验评分（**缺失约 35%**） |
| `review_scores_communication` | float64 | 3027 | 7 | 沟通评分（缺失约 35%） |
| `review_scores_location` | float64 | 3028 | 7 | 位置评分（缺失约 35%） |
| `review_scores_rating` | float64 | 3036 | 36 | 总评分，量纲是 0~100（缺失约 35%） |
| `review_scores_value` | float64 | 3027 | 8 | 性价比评分（缺失约 35%） |
| `room_type` | object | 4667 | 3 | 房型，如 Entire home/apt、Private room |
| `rental_price_USD` | int64 | 4667 | 428 | **目标**：日租美金价格 |

`test-ans.xlsx`（1167 行 × 27 列，字段完全一样，练习这份带着答案列）：
`bathrooms` 1156 非空、`beds` 1162、`host_is_superhost` 1165、
五个 `review_scores_*` 都是 ~750 非空（缺失比例和训练集接近）。

几个一眼要记住的点：

- `has_availability` 只有一个取值 `t`，**留着就是白占一列**。
- 5 个 `review_scores_*` 缺失都接近三分之一，不是零星缺失，怎么填会明显影响结果。
- `amenities` 平均每行十几个设施名，全表 3666 个不同字符串 —— 它是个多值字段，不是类别字段。
- `rental_price_USD` 值域跨得很大（`maximum_nights` 能到 1125、价格也有极端高值），
  目标右偏，这是后面要处理的东西。

## 三、思路

### 1. 先按字段的「性质」分四组，别一锅端

```python
category_cols = ['bed_type', 'cancellation_policy', 'property_type', 'room_type']
binary_cols   = ['has_availability', 'host_is_superhost', 'instant_bookable']
numeric_cols  = list(set(train_X.columns) - set(category_cols) - set(binary_cols) - {'amenities'})
```

- **分类列**（4 个）→ `pd.get_dummies`。
- **布尔列**（3 个）→ `t`/`f` 换成 1/0。
- **数值列**（17 个）→ 缩放 + 缺失填充。
- **`amenities`** 单独走一条路：拆成设施清单 → multi-hot → 降维。

这个分法不是我拍的，是照着 `description.xlsx` 那张「欄位特性」表来的。

### 2. `amenities` 是本题特征工程的主战场

它一行就是十几个设施，直接 `LabelEncoder` 会得到「几乎每行一个编号」的废特征
—— 3667 行里 3666 个唯一值，等于把 `room_id` 编码了一遍。

我第一版（`ans.ipynb`）的做法是：
拆成字典 → `DictVectorizer` 做 multi-hot → `PolynomialFeatures(degree=3)` 造组合
→ `PCA` 压到 14 维 → 拼回主表。

后来改进版（`templates/04_回归_改进版.py`）换成了更直接的做法：
取高频设施各给一个 0/1 特征（前 60 个）+ 一个设施数量 `len_amenities`。
理由是 PCA 之后的 14 个主成分没法解释，而「有没有空调/泳池」这种特征是能讲清楚的。

### 3. 目标右偏 → log1p

价格这种量纲天然右偏（模板注释里记的：中位数 175、最大值 10000，差两个数量级）。
改进版加了 `np.log1p` 变换，
训练完再 `expm1` 还原 —— 这是回归题的常规操作，代价是 RMSE 会在大值上被放大，
不过对排名通常有好处。

### 4. 模型

- `ans.ipynb`：`RandomForestRegressor`，全部默认参数，holdout 0.2 单次划分。
- 改进版：`LGBMRegressor(objective='regression_l1')` + 5 折 KFold + 早停，
  记录 `best_iteration_`，最后用平均最佳轮数 × 1.1 全量重训一遍出提交文件。

## 四、关键代码

**① 删掉近常数的哑变量**

```python
t = pd.get_dummies(train_X, columns=category_cols, prefix='category')
mean_series = t.mean(numeric_only=True)
[idx for idx, v in mean_series.items() if v < 0.000215]
# → ['category_no_refunds', 'category_Earth House', 'category_Hut', 'category_Tipi']

train_X = pd.get_dummies(train_X, columns=category_cols, prefix='category')
train_X.drop(['category_no_refunds', 'category_Earth House', 'category_Hut', 'category_Tipi'],
             axis=1, inplace=True)
```

`property_type` / `cancellation_policy` 里有几个长尾类别，出现比例低于 0.000215（万分之 2 左右），
one-hot 出来就是「4667 行里 1 行是 1」的列。这种列树模型基本学不到东西，
还会让特征重要性表变脏，索性删掉。

**② `t`/`f` 转 0/1 + 数值列缩放**

```python
train_X[binary_cols] = np.where(train_X[binary_cols] == 't', 1, 0)

MMScaler = MinMaxScaler()
train_X[numeric_cols] = MMScaler.fit_transform(train_X[numeric_cols])
```

`np.where(df == 't', 1, 0)` 一行同时处理三列，比逐列 `map` 干净。
`has_availability` 全表都是 `t`，所以它转换完就是全 1 的常数列 —— 等于没删但也没害。

**③ `amenities` 拆成字典**

```python
def preprocess(s_):
    s_ = s_.replace('{', '').replace('}', '')
    s = {item: 1 for item in s_.split(',') if item != ''}
    return s

train_X['amenities'] = train_X['amenities'].apply(lambda x: preprocess(x))
train_X['len_amenities'] = train_X['amenities'].apply(lambda x: len(x))
```

字符串本身长得就像 Python 的 set 字面量，只是外层是大括号。
手写 `replace` 拆比 `ast.literal_eval` 稳（有 4667 行里那一条空字符串 `{}` 的边界情况）。
顺便造了个 `len_amenities` —— 设施数量本身就是个不错的「房源档次」代理变量。

**④ multi-hot → 多项式 → PCA**

```python
dictvectorizer = DictVectorizer(sparse=False)
features = dictvectorizer.fit_transform(train_X['amenities'])
amenities_df = pd.DataFrame(features, columns=dictvectorizer.get_feature_names_out())

pf = PolynomialFeatures(degree=3, include_bias=False)
poly_X = pd.DataFrame(pf.fit_transform(amenities_df), columns=pf.get_feature_names_out())

pca = PCA(n_components=14, random_state=2023)
pca_X = pd.DataFrame(pca.fit_transform(poly_X),
                     columns=[f'pca{x}' for x in range(14)], index=train_X.index)
concat_X[[f'pca{x}' for x in range(14)]] = pca_X
```

三段都做在 `amenities` 这一路上，最后只往主表加 14 列。
好处是维度可控；坏处是 `fit_transform` 之后我看不出哪个主成分对应什么设施。
`index=train_X.index` 这步不能省 —— PCA 丢掉了原始索引，不指定就会拼错行。

**⑤ 改进版：目标 log 变换 + LGBM 早停**

```python
yy = np.log1p(y) if USE_LOG else y
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr_i, va in kf.split(X):
    m = LGBMRegressor(objective='regression_l1', n_estimators=4000, learning_rate=0.02,
                      num_leaves=31, colsample_bytree=0.8, subsample=0.8,
                      reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr_i], yy[tr_i], eval_set=[(X.iloc[va], yy[va])],
          callbacks=[early_stopping(100, verbose=False)])
    oof[va] = m.predict(X.iloc[va])
    test_pred += m.predict(X_test) / 5
```

`objective='regression_l1'` 是刻意的：目标右偏严重时，L1 比默认的 L2 更抗离群点。
用 5 折的平均预测当测试集结果，比单次 holdout 稳。

**⑥ 写出提交文件**

```python
result = pd.DataFrame({ID_COL: test_ids, OUT_COL: pred})
result.to_csv('考生姓名_results.csv', index=False)
print('列名:', list(result.columns), '  ← 必须是 room_id 和 predicted_rrental_price_USD')
```

打印列名这一步是我加的「提交前自检」—— 就是被双 r 坑过一次之后加的。

## 五、踩的坑

**1）`predicted_rrental_price_USD` 是双 r。**
我第一版写成了 `predicted_rental_price_USD`（单 r），
模板文件的注释里把这条列为「致命·格式」：题目原文就是这么拼的，照着抄。
写错的后果是格式错误 → 0 分，跟 RMSE 多少完全无关。

**2）测试集的缩放器是重新 `fit` 的。**

```python
MMScaler = MinMaxScaler()
train_X[numeric_cols] = MMScaler.fit_transform(train_X[numeric_cols])
...
test_X[numeric_cols] = MMScaler.fit_transform(test_X[numeric_cols])   # 又 fit 了一次
```

第二行本意应该是 `MMScaler.transform(...)`。重新 `fit` 意味着训练集和测试集
各自按自己的 min/max 归一化，同一个数值在两边代表不同的东西 ——
树模型对这个不敏感，但如果换成线性模型结果会很难看。
正确写法永远是 **fit 在训练集、transform 到测试集**。

**3）缺失的近三分之一，被 `fillna(0)` 变成了「评分为 0」。**

```python
y_tst_hat = reg.predict(concat_tstX.fillna(0))
```

`review_scores_*` 缺失率约 35%，填 0 相当于告诉模型
「这个房源入住体验 0 分、性价比 0 分」。要是先缩放到 [0,1] 再填 0，
0 刚好是缩放后的最小值，语义就更歪了。
改进版换成中位数填充（`fillna(full[c].median())`），这个是明显更安全的默认值。

**4）训练/测试列对齐那段写反了。**

```python
missing_columns = existing_columns.difference(concat_tstX.columns)
for col in missing_columns:
    concat_X[col] = 0        # 本该补到 concat_tstX 上
extra_columns = concat_tstX.columns.difference(existing_columns)
concat_tstX.drop(extra_columns, axis=1, inplace=True)
```

`concat_X` 本来就有这些列，给它补 0 是空操作；真正缺列的 `concat_tstX` 却没补上。
这次能跑通说明当时 `missing_columns` 恰好是空的，但这段逻辑本身是错的，
下次遇到测试集独有类别（这次就有 `category_Boat`、`category_Chalet` 被 drop 掉了）
迟早要翻车。

**5）单次 holdout 估分不稳。**
`ans.ipynb` 用 `train_test_split(test_size=0.2, random_state=0)` 报了 **RMSE 222.91**，
同一套特征在完整测试集上是 **RMSE 314.20** —— 这两个数都是 notebook 的输出里真实留下的。
差了九十多，说明那一折恰好偏容易。改成 K 折 OOF 才能拿到可信的估分。

**6）`room_id` 不能进特征。**
`ans.ipynb` 用 `index_col='room_id'` 读进来，改进版显式写了
`feat_cols = [c for c in train_df.columns if c not in (TARGET, ID_COL)]`。
一旦让模型看到序号，它会去记「第 5833 号房源贵」，毫无泛化。

**7）`description.xlsx` 里的 `欄位` 是带空格的。**
那张说明表写作 `availability 30` / `cancellation policy`，
真实列名却是 `availability_30` / `cancellation_policy`。
拿说明表去对列名的时候别直接字符串匹配，看一眼真实 dtype 更省事。

## 六、小结

这题我最大的收获是**「字段性质分类」这一步要单独做，而且要写在代码里**：

- 4 个分类列、3 个布尔列、17 个数值列、1 个多值列表列 ——
  四种处理方式完全不同，混着来一定出错。
- 多值字段（`amenities`）的价值远大于它的列数：4667 行里 3666 个唯一值，
  说明每行的组合都不一样，这里的差异才是价格差异的来源之一。
- 缺失率要看**比例**再决定填法：零星缺失随便填，缺三分之一就必须想语义。

如果再跑一次，我会先做两件小事：把 `fillna(0)` 换成中位数、
把 `test` 的缩放改成 `transform`，然后才谈调参。
