# 07 AI 生成文本识别

> 判断一段文字是人写的还是 AI 写的 —— 而真正决定分数的不是语义模型，是 `length` 这一列元数据。

| | |
|---|---|
| 题型 | 文本 · 二分类（True / False） |
| 数据 | 训练 4000 行 × 4 列，测试 1000 行 × 4 列 |
| 提交 | `Final_Results.csv`，列：`id`, `Predicted_Result` |
| 评分 | Accuracy |
| 代码 | [`code/07-ai-text-detect`](../code/07-ai-text-detect) · [`templates/07_文本二分类.py`](../templates/07_文本二分类.py) |

## 一、题目在问什么

一句话：**给一篇文章，判断它是人写的，还是 ChatGPT 这类模型生成的。**

目标列是 `real`，布尔类型：

- `True` = 人写的
- `False` = AI 生成的

评分用 Accuracy（正确率）。这个口径跟前面几篇都不一样 —— 01 看负评 F1、02 看 Macro-F1、05 看 Micro-F1，而这一篇**只看整体正确率**。这意味着：

- 类别均衡的话，Accuracy 是合理的；
- 但只要有一类占绝对多数，就可以靠「全猜多数类」白拿分。

题面还有个硬门槛：正确率**不到 75% 直接 0 分**。所以这题的关键不是「把模型调精」，而是「保证稳稳过线」—— 这个心态差别很大，后面会看到它是怎么影响我的技术选择的。

背景动机题面写了一堆（假新闻、诈骗、社交工程），跟建模没什么关系，跳过。

## 二、数据长什么样

两个 CSV：

| 列名 | 类型 | unique（训练/测试） | 说明 |
|---|---|---|---|
| `id` | int64 | 4000 / 1000 | 文章序号，提交时的 ID 列 |
| `text` | object | 4000 / 1000 | 文章正文，**全部唯一，没有重复** |
| `length` | int64 | 935 / 490 | 文章词数，**范围 43 ~ 1024** |
| `real` | bool | 2 / 2 | **目标列**，True = 人写，False = AI 生成 |

训练集 4000 行 × 4 列，测试集 1000 行 × 4 列，字段完全一致。

### 这份数据里藏着一个致命信号

真正让我在这次练习里学到东西的，是 `length` 这一列的分布。答案在题面之外的地方 —— 我后来在方法论文档里记下来了：

> **AI 文本恒 = 1024 词。**

也就是说，`length` 的最大值 1024 这个「整数上界」本身就是 AI 生成文本的标记。AI 生成的样本全都被截到 1024 词（生成时的长度上限），而人写的文章长度是自然分布，随机散落在 43~1024 之间。

这决定了整个练习的性质：**这道题的分数不在语义上，在 `length` 这一个数字上。** 特征是王道，不是模型。

我当时并不知道这一点（是后来复盘时才发现 `length` 列的上界这么整齐），但我的代码**恰好把 `length` 当特征喂进去了**。这一步是运气好，但回头看，正确的说法是：**任何数值列，不管看起来多无聊，都应该先画个直方图看看。**

## 三、思路

这份 notebook 只有 8 个 cell，是四个练习里最短的。流程极简：

```text
读数据 → 拆出 y → 丢掉 id / real → TF-IDF(text) + length → 决策树 → 写 CSV
```

逐个说决策：

**1. 文本向量化用最基础的 `TfidfVectorizer()` 默认参数。**
没调 `ngram_range`、没调 `min_df`、没加字符级通道。理由很简单：这题的时间预算有限，先把 pipeline 跑通。

**2. 把 `length` 作为额外特征拼到 TF-IDF 矩阵上。**
这是整份代码里最有价值的一行（第 29 / 32 行）。做法是 `scipy.sparse.hstack` 把稠密的一列拼到稀疏矩阵后面。

**3. 模型用 `DecisionTreeClassifier()`，默认参数。**
朴素得不能再朴素。用它的原因不是它准，而是：

- 决策树**对特征尺度不敏感**，可以直接吃未归一化的 `length`；
- 它**几秒钟就能跑完**，符合「先拿到提交文件」的原则；
- 代码里同时还 `import` 了 `RandomForestClassifier`（注释掉的），说明我当时想的是「先用树跑通，再换森林」。

**4. 用 5 折交叉验证先估一下泛化能力，再拟合全量。**
`cross_validate` 默认 `cv=5`，同时算 `accuracy` 和 `f1` 两个指标。

## 四、关键代码

### 4.1 拆出标签，然后一定要把 `real` 从特征里删掉

```python
X_train = pd.read_csv("../data/training.csv")
X_test  = pd.read_csv("../data/test-ans.csv")

y_train = X_train.real
y_test  = X_test.real
X_test_id = X_test.id

X_train.drop(columns=['id', 'real'], inplace=True)
X_test.drop(columns=['id', 'real'], inplace=True)
```

**`drop(columns=['id', 'real'])` 这一步是整份代码里最关键的三行。**

`real` 是答案列，留在特征里就是 100% 的标签泄漏。这种错误在 notebook 里很常见：`y = df.real` 之后忘了 `drop`，模型秒到 1.0，自己还挺高兴。

`id` 也一起删了 —— 它是唯一标识，没有任何泛化信号，留着只会让模型背下训练集的 ID。**提交用到的 ID 要单独存一份**（`X_test_id`），但绝不能进特征矩阵。

这里的 `X_train` / `X_test` 变量名起得挺误导的（它们是「特征 DataFrame」不是「训练/测试矩阵」），后面 `X_train_vect` 才是向量化后的。看代码时要小心区分。

### 4.2 把单列数值特征拼到稀疏矩阵上

```python
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

vect = TfidfVectorizer()
X_train_vect = vect.fit_transform(X_train.text)
X_train_vect = add_feature(X_train_vect, X_train.length)

X_test_vect = vect.transform(X_test.text)
X_test_vect = add_feature(X_test_vect, X_test.length)
```

三个技术点：

1. **`csr_matrix(...).T`** —— `X_train.length` 是一维的 Series，要变成 `(n, 1)` 的列向量才能横向拼接，所以取转置。直接 `hstack` 一维数组会报维度错或者拼出奇怪的结果。
2. **`hstack(..., 'csr')`** —— 稀疏矩阵用 `scipy.sparse.hstack`，不是 `np.hstack`。TF-IDF 出来的矩阵可能有几万到几十万维，用 `np.hstack` 会直接 OOM。
3. **测试集只 `transform`，不重新 `fit`** —— 这行写得对，向量器是训练集上 fit 的那个 `vect`。

**唯一的问题：`length` 没有归一化。** `length` 的量级是 43~1024，而 TF-IDF 的值域是 0~1 左右。对决策树来说无所谓（树按阈值分裂，不看尺度），但如果后面换成逻辑回归或 SVC，这一列会把所有 TF-IDF 特征压成噪声。模板里后来补上了这一步：

```python
scaler = MinMaxScaler().fit(np.concatenate([train['length'], test['length']])
                            .astype(float).reshape(-1, 1))
len_tr = sparse.csr_matrix(scaler.transform(train['length'].astype(float).values.reshape(-1, 1)))
A = sparse.hstack([A, len_tr]).tocsr()
```

**「数值列拼进稀疏特征前先归一化」，这条对线性模型是必须的，对树模型是可选的。**

### 4.3 先交叉验证，再全量拟合

```python
cv_results = cross_validate(
    DecisionTreeClassifier(),
    X_train_vect,
    y_train,
    scoring=["accuracy", "f1"],
    cv=10,
)
print('Average Accuracy: {0} +/- {1}'.format(cv_results['test_accuracy'].mean(),
                                            cv_results['test_accuracy'].std()))
print('Average F1: {0} +/- {1}'.format(cv_results['test_f1'].mean(),
                                      cv_results['test_f1'].std()))
```

这里的 `cv=10` 是 **10 折交叉验证**（`cross_validate` 的默认值是 5，我显式调大了）。4000 行数据跑 10 折，用默认参数的决策树依然几秒钟就完事 —— 这也是我敢用交叉验证而不是单次切分的原因。

`scoring` 传列表可以一次算多个指标。**`std`（标准差）比 `mean` 更值得看** —— 如果 10 折之间波动很大，说明模型不稳定，均值再高也不可靠。

### 4.4 输出：注意列名和索引

```python
y_pred = model.predict(X_test_vect)
df = pd.DataFrame(data={'id': X_test_id, 'Predicted_Result': y_pred}).set_index('id')
df.to_csv('Final_Results.csv')
```

两个细节：

1. **`set_index('id')` 之后又 `to_csv()` 没加 `index=False`。** 效果是 ID 被写成了 CSV 的索引列，正好是想要的两列格式。但这属于「绕了一圈达到目的」—— 正常写法是 `to_csv('Final_Results.csv', index=False)`，把 `id` 当普通列。前者万一不小心多一层 `reset_index` 就会多出一列。
2. **列名是 `Predicted_Result`（单数）**，不是前几篇的 `Predicted_Results`（复数）。**提交文件的列名差一个字母，就是拿分和不拿分的区别**，每个练习都得逐字核对。

另外，`y_pred` 是 numpy 的 `bool_` 数组，写进 CSV 会变成 `True` / `False` 字符串 —— 这正好符合题面「必须是 True/False，不可为数字」的要求。这份文件在数据卡片里读回来确实是 bool 类型，说明格式没问题。

## 五、踩的坑

### 坑 1：默认参数的决策树会严重过拟合

`DecisionTreeClassifier()` 默认不限制深度，在几万维的 TF-IDF 上会一路长到把每个训练样本都单独分出来。**在训练集上 100%，在测试集上掉一大截** —— 这是决策树的经典行为。

正确做法是至少限制一下：

```python
DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=0)
```

或者干脆换成随机森林（代码里那行注释掉的 `RandomForestClassifier(random_state=0)` 就是为这个准备的）。**「默认参数的树」是能跑但不可信的东西**，交叉验证的 `std` 大不大，就是在提示这件事。

### 坑 2：`real` 是 bool，不是 0/1

```python
y_train = X_train.real      # dtype = bool
```

sklearn 能直接吃 bool 标签（内部当作 0/1），`accuracy_score` 和 `f1_score` 也都能处理。但**跨数据集、跨文件的时候要小心**：

- `pd.read_csv` 把 `True`/`False` 读成 bool，而 `0`/`1` 会读成 int64，两者混在一起会比较出意料之外的结果；
- 题目要求提交 `True`/`False`，如果你中途把标签转成了 `0/1`，最后一定要转回去（用 `LabelEncoder.inverse_transform` 或者 `map({0: False, 1: True})`）。

模板里的做法是统一先 `astype(str)` 再 `LabelEncoder().fit_transform()`，最后 `inverse_transform` 回来：

```python
le_target = LabelEncoder()
train[TARGET] = train[TARGET].astype(str).fillna('')
y = le_target.fit_transform(train[TARGET])
...
pred_label = le_target.inverse_transform(pred)
```

`astype(str)` 把 `True` 变成字符串 `'True'`，`LabelEncoder` 排完序（`'False'` < `'True'`）得到 `{False: 0, True: 1}`，最后逆变换回 `True`/`False`。**这样任何类型的标签（bool / 字符串 / 中文类别名）都能走同一条路。**

### 坑 3：`length` 是泄漏特征 —— 这既是好事也是坏事

**好事**：分数主要来自这一列。方法论文档里记的结论是，用 `length` 这类元数据泄漏特征，accuracy 能从 0.90 提到接近 1.0；模板文件顶部记的对照是 0.926 → 0.939（这是同一个练习在两代模板上的实测差异）。

**坏事**：这意味着模型学到的是「AI 输出被截断到 1024 词」这个**数据构造artifact**，不是「AI 文风」。换一个不做长度截断的数据集，这个模型立刻失效。

这两个数字都来自仓库里的模板/文档记录，不是这份 notebook 跑出来的 —— 这份 notebook 的 cell 里没有留任何输出结果，所以**我无法告诉你它当时的交叉验证分数是多少**。

我做错的地方是：**当时没有任何「去 feature 消融」的动作**。正确做法很简单：

```python
# 对照实验：去掉 length，只用 text
X_ablation = add_feature(vect.transform(...), ...)  # 不拼 length
```

跑一遍不带 `length` 的版本，两个 accuracy 一对比，就知道分数到底来自语义还是来自元数据。**「这个分数是谁贡献的」应该靠消融实验回答，不是靠猜。** 这是我在这份练习里最该补上、却完全没做的一步。

### 坑 4：只有 `DecisionTreeClassifier` 一个模型，没有基线对照

代码里 `import` 了 `RandomForestClassifier` 但注释掉了，全程只跑了一棵树。**没有基线，就不知道这棵树到底算好还是算差。**

后来模板里的做法是先算多数类基线：

```python
lab, cnt = np.unique(y, return_counts=True)
baseline = score(y, np.full(len(y), lab[cnt.argmax()]))
print(f'多数类基线分 : {baseline:.4f}')
if best_s <= baseline:
    print('⚠ 没打过多数类基线，建议直接全填多数类交卷，或回头查特征。')
```

一句 `np.full` 就能算出「全猜多数类」的分数。**任何分类任务的第一个动作都应该是算这个数** —— 打不过它，说明整个特征工程都是白做的。这个练习的 75% 及格线更是如此：先确认多数类基线是不是已经过线了。

### 坑 5：不要用 `cross_val_predict`

这份代码里用的是 `cross_validate`（正常）。但后来在别的练习里我用 `cross_val_predict` 拿 OOF 概率调阈值，结果撞上了版本坑：`xgboost 2.0.3 + sklearn 1.6.1` 下会抛 `AttributeError: 'super' object has no attribute '__sklearn_tags__'`。

模板里的绕法是**手写 OOF 循环**：

```python
def oof_proba(m):
    """手写 out-of-fold，绕开 cross_val_predict 的 xgboost/sklearn 版本 bug。"""
    oof = np.zeros((len(y), len(classes)))
    for tr_i, va_i in cv.split(A, y):
        m.fit(A[tr_i], y[tr_i])
        oof[va_i] = m.predict_proba(A[va_i])
    return oof
```

十几行代码，不依赖 sklearn 内部实现，换版本也不会崩。**依赖库版本 bug 这种东西，绕过去比修它划算。**

## 六、小结

这份练习是四个里面代码最短、但教训最实在的一个。因为它的分数**根本不是模型挣来的**：

1. **`drop(columns=['id', 'real'])` 是保命的三行。** 答案列留在特征里，模型秒到满分，然后你什么也没学到；
2. **数值列（如 `length`）值得先画个直方图。** 我就是因为多看了一眼这列，才在后来发现「AI 文本恒 = 1024」这个上界 —— 元数据泄漏往往比语义特征更值钱；
3. **数值列拼进稀疏特征前要归一化**（用 `MinMaxScaler`），否则线性模型会被量级碾压；
4. **默认参数的决策树会过拟合**，至少要给 `max_depth` / `min_samples_leaf`；
5. **先算多数类基线**，打不过就别谈调模型；
6. **去 feature 消融是必须做的**，否则你永远不知道该谢特征还是谢模型；
7. **标签类型（bool / 0-1 / 中文名）统一走 `LabelEncoder` + `inverse_transform`**，一条路通吃，不容易在输出格式上翻车。

最有价值的一条心法是方法论文档里那句话：**判题型看「输出 + 评分」，找分数看「泄漏特征 + 信号源」，不盲目堆模型。**

这道题的评分只看准确率，及格线 75%。而现在回头看，正确的技术路线应该是：

```python
# 1. 先算多数类基线，确认是否已经过线
# 2. char_wb + word 双通道 TF-IDF（train+test 一起 fit）
# 3. length 归一化后 hstack 进去
# 4. XGBoost 浅树少树：n_estimators=100, max_depth=4
# 5. 手写 OOF 选模型，按 accuracy 选
```

这套配置在模板文件顶部记的实测结果是 **accuracy 0.939**（对照旧版模板的 0.926）。同样也从模板里学到一条反直觉的结论：**XGBoost 用「浅树 + 少树」（n=100, d=4）比「深树 + 多树」（n=300, d=6）又快又准**（38 秒 0.939 vs 148 秒 0.936）。**堆模型规模不一定涨分，先把特征做对。**

一句话总结：**这道题赢在 `length` 那一列上，而我当时只是碰巧把它喂了进去 —— 下次要变成有意识地去翻每一列。**
