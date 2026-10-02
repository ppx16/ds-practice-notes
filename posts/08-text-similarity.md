# 08 文本对相似度

> 这道题真正的对手不是模型，是「全预测 1」这个平凡基线；
> 另外它是**二分类**（判断两句话是否同义），不是回归 —— 目标只有 0 和 1。

| | |
|---|---|
| 题型 | 文本 · 二分类（句对相似度判定） |
| 数据 | 训练 3,260 行 × 4 列，测试 816 行 × 3 列 |
| 提交 | `<姓名>_results.csv`，列：`Pair_Num`, `Predicted_Result` |
| 评分 | F1（正类 = 1） |
| 代码 | [`templates/08_文本对相似度.py`](../templates/08_文本对相似度.py) |

> 「代码」这一栏只有一个模板，是因为这个练习**没有 `code/08-text-similarity` 目录** ——
> 仓库里唯一的产物就是上面那个骨架，所以下面的片段全部出自它，
> 我也给不出「某次运行的分数」以外的实验记录。

## 一、题目在问什么

给一堆**成对的句子**，判断这两句是不是在说同一件事。
训练数据每行是一对句子加一个 `Similar` 标签：1 表示语义相似，0 表示不相似；
测试数据给出 816 对句子，要我对每一对输出 0 或 1。

提交两列：`Pair_Num`（配对编号）和 `Predicted_Result`（0/1 整数）。

评分只看**正类（=1）的 F1**，不是准确率，也不是 macro-F1。
这个细节决定了整道题的策略 —— 后面会看到它有多重要。

我手里没有这道题的原始题面文件，上面的描述是从数据卡片和模板代码反推的：
`Pair_Num` / `Sentence1` / `Sentence2` / `Similar` 四个字段、输出 0/1、评分是正类 F1，
这几点在数据卡片、模板文件头注释和方法论文档里是一致的。

## 二、数据长什么样

`Training.xlsx`（3260 行 × 4 列）：

| 列名 | 类型 | 非空 | unique | 说明 |
|---|---|---|---|---|
| `Pair_Num` | int64 | 3260 | 3260 | 配对编号，主键（模板里当索引读） |
| `Similar` | int64 | 3260 | 2 | **目标**：1 相似 / 0 不相似 |
| `Sentence1` | object | 3260 | 3260 | 第一句，全是英文新闻句 |
| `Sentence2` | object | 3240 | 3138 | 第二句，**有 20 条缺失**，且本身有重复 |

`Test.xlsx`（816 行 × 3 列）：

| 列名 | 类型 | 非空 | unique | 说明 |
|---|---|---|---|---|
| `Pair_Num` | int64 | 816 | 816 | 配对编号（从 4000 起） |
| `Sentence1` | object | 816 | 816 | 第一句 |
| `Sentence2` | object | 805 | 798 | 第二句，**有 11 条缺失** |

交出去的 `考生姓名_results.csv` 是 816 行 × 2 列：`Pair_Num`（和测试集一一对应）
和 `Predicted_Result`（int64，只有 0/1 两个取值）。

从上面几张表的 `unique` 列能看出几件事：

- 3260 行里 `Sentence1` 有 3260 个唯一值，`Sentence2` 只有 3138 个 ——
  **第二句会被复用**，同一个句子会和不同的第一句配对。
- 缺失落在 `Sentence2` 上（训练 20 条、测试 11 条），`Sentence1` 是齐的。
- 标 1 的基本都是「改写句」：换了语序、换了从句结构、也换了不少用词，但意思不变。
- 标 0 的往往是**数字或事实对不上**，比如都提到同一只股票在涨，
  但一个说涨 19 美分、另一个说涨 20 美分。这类「数字冲突」最容易翻车。

## 三、思路

### 1. 先判题型，再算基线

目标是 0/1，评分是正类 F1 → **二分类**。
而模板注释里写着：正类占 **67.5%**，
所以「全部预测 1」这个什么都不做的基线，F1 就有 **0.8063**。
模板干脆把这条写成了一个警告：

```python
base_f1 = f1_score(y, np.ones_like(y))
if max(scores) <= base_f1:
    print('⚠ 模型没打过"全预测1"，建议直接全填 1 交卷，或回头查特征。')
```

**这就是本题的第一性原则：先算出平凡基线，一切以「有没有打过它」为准。**

### 2. 手工造一组互补的相似度特征

单个相似度指标顶不住，所以模板铺了一组视角不同的特征，合起来 17 维：

| 特征 | 视角 | 直觉 |
|---|---|---|
| `cos_word` | 词级 TF-IDF 余弦（1~2 gram） | 用词重合度 |
| `cos_char` | 字符级 TF-IDF 余弦（char_wb 2~4） | 拼写/词形变化容忍 |
| `jaccard` | 词集合交并比 | 重合词的绝对比例 |
| `contain` | 交集 / 较短那句 | 一句包含另一句 |
| `prec` / `rec` | 交集 / 各自长度 | 两个方向的重合率 |
| `jac_nostop` | 去停用词后的 Jaccard | 去掉 the/a/of 的干扰 |
| `jac_bigram` | 词二元组 Jaccard | **词序**信息 |
| `seqratio` | `difflib` 字符序列相似度 | 改动幅度 |
| `lcs` | `difflib` 词序列相似度 | 语序重排 |
| `len_diff` / `len_ratio` / `len_mean` | 长度差 / 比 / 均值 | 长度极不匹配基本不相似 |
| `only1` / `only2` | 各自独有的词数 | 谁多说了什么 |
| `num_jac` / `num_diff` | 数字集合的交并比 / 差异数 | **专门抓数字冲突** |

`num_jac` / `num_diff` 是我最喜欢的一对：新闻改写题里，
意思相同的句子用词可以差很多，但**数字必须一致**。
把两句话里的数字都抽出来比一比，能直接拆掉一批「看着很像但事实不同」的假阳性。

### 3.（可选）加一层语义 embedding

TF-IDF 抓的是字面重合，遇到「换个词说同一件事」就没辙。
所以模板留了一档更重的做法：用 `bge-m3` 编码，加两个特征 ——
句向量余弦，以及 `|e1 - e2|` 降维后的 32 维。
设计上是**两档自动切换**：找得到本地模型就走「满血档」，
找不到就退回纯 TF-IDF 的「离线档」，不会因为缺模型直接中断。

### 4. 阈值不能取 0.5

评分只看正类 F1、正类又占 67.5% —— 这意味着分类器**偏向预测 1**通常更划算。
所以模板在 0.05~0.95 上网格搜一遍阈值，取使正类 F1 最大的那个。

## 四、关键代码

**① 清洗：两列都要补缺失**

```python
for d in (train, test):
    d[COL1] = d[COL1].fillna('')
    d[COL2] = d[COL2].fillna('')

def clean(t):
    """小写化 + 只保留 字母/汉字/数字/空格。给 TF-IDF 用。"""
    return re.sub(r'[^a-zA-Z一-鿿0-9 ]', ' ', t.lower()) if isinstance(t, str) else ''
```

`for d in (train, test)` 一次处理两个表。只保留字母/汉字/数字/空格，
是为了让标点和大小写不干扰字符 n-gram。

**② 逐行余弦相似度：向量化写法**

```python
def cos_rows(a, b):
    # TfidfVectorizer 默认 norm='l2'，向量已单位化 -> 余弦 == 点积
    # 比 for 循环调 cosine_similarity 快 15 倍
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()
```

这是整份模板里我最想记住的一招：TF-IDF 向量已经是 L2 归一化的，
所以逐行点积就是余弦，不需要构造完整的 3260×3260 相似度矩阵。
模板注释还留了一条：旧版把余弦和点积当成两个特征，实测相关系数 **1.000000**，纯重复。

**③ 特征工程主体（节选）**

```python
f['cos_word'] = cos_rows(WORD.transform(df['c1']), WORD.transform(df['c2']))
f['cos_char'] = cos_rows(CHAR.transform(df['c1']), CHAR.transform(df['c2']))

S1 = [set(a.split()) for a in s1]
S2 = [set(b.split()) for b in s2]
f['jaccard'] = np.array([len(a & b) / max(len(a | b), 1) for a, b in zip(S1, S2)])
f['contain'] = np.array([len(a & b) / max(min(len(a), len(b)), 1) for a, b in zip(S1, S2)])
f['jac_bigram'] = np.array([len(a & b) / max(len(a | b), 1) if (a or b) else 0.
                            for a, b in zip(B1, B2)])   # B1/B2 是词二元组集合
```

集合运算全部用 `&` / `|`，配合 `max(..., 1)` 防除零 ——
两句话里至少有一句可能被 `clean` 洗成空串，这个保护不能省。

**④ 数字特征要从「原文」抽，不能用清洗后的**

```python
NUM = re.compile(r'\d+\.?\d*')
n1 = [set(NUM.findall(str(t))) for t in df[COL1]]
n2 = [set(NUM.findall(str(t))) for t in df[COL2]]
f['num_jac']  = np.array([len(a & b) / max(len(a | b), 1) if (a or b) else 1.
                          for a, b in zip(n1, n2)])
f['num_diff'] = np.array([len(a ^ b) for a, b in zip(n1, n2)], float)
```

注意这里是 `df[COL1]` 而不是洗过的 `df['c1']`。模板注释写得很清楚：
「`clean` 会破坏 `$` `%` `.` 等符号」—— `$2.11` 洗成 `2 11` 之后小数点和美元号都没了。
`len(a ^ b)` 是对称差，正好是「两边数字集合不一致的个数」。

**⑤ 两档自动切换**

```python
def find_model(name):
    """在当前目录及上级目录里找 models/<name>。找不到返回 None。"""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        p = os.path.join(d, 'models', name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'config.json')):
            return p
        d = os.path.dirname(d)
    return None
```

向上找 4 层目录，用 `config.json` 是否存在判断「这个文件夹真的是个模型」；
返回 `None` 时打印一行提示继续跑。**这个降级设计比多 1 个点的分数重要**，
因为出结果比出好结果优先。

**⑥ 训练 + 用 OOF 概率搜阈值**

```python
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]

grid = np.arange(0.05, 0.96, 0.01)
scores = [f1_score(y, (oof >= t).astype(int)) for t in grid]
BEST_T = float(grid[int(np.argmax(scores))])
```

一定要用**交叉验证的 out-of-fold 概率**选阈值：在训练集上
`clf.predict_proba(X)` 再搜阈值，就是在自己卷子上找答案，搜出来会虚高。
`make_pipeline` 也保证缩放器只在每折的训练部分 fit，不漏验证折的信息。

**⑦ 输出**

```python
clf.fit(X, y)
pred = (clf.predict_proba(X_test)[:, 1] >= BEST_T).astype(int)
pd.DataFrame({ID_COL: test.index, 'Predicted_Result': pred}).to_csv(OUT_CSV, index=False)
print(f'预测为1比例: {pred.mean():.3f}  (训练集真实比例 {y.mean():.3f})')
```

最后那行打印是自检：预测为 1 的比例和真实比例差太远，说明阈值搜歪了。

## 五、踩的坑

**1）最大的坑是「没算平凡基线就埋头调模型」。**
模板注释里那四行数字是我最该先看的（都是模板文件里记录的实测值，不是我这次跑出来的）：

| 方案 | 正类 F1 |
|---|---|
| 全预测 1 的平凡基线 | **0.8063** |
| 旧版模板（TF-IDF cos + dot） | 0.8026 |
| 本版离线档（纯 TF-IDF，17 维） | 0.8295 |
| 本版满血档（+ bge-m3，50 维） | 0.8446 |

（口径是 5 折 CV × 5 个随机种子，标准差都在 0.0008 左右。）

注意第二行：**旧版模板 0.8026，比什么都不做还低**。
不知道基线在哪，就会以为自己做出了一个「还行的模型」，实际上是从零开始。

**2）两列都要补缺失，只补一列会漏。**
`Sentence2` 训练缺 20 条、测试缺 11 条，`Sentence1` 是全的。
一开始只写了 `d[COL1] = d[COL1].fillna('')`，清洗那一步碰到 NaN 就炸。

**3）`0.5` 不是本题的好阈值。**
评分只看正类 F1、正类又占 67.5%，所以最优阈值通常低于 0.5。

**4）`|e1 - e2|` 直接上 1024 维会过拟合。**
bge-m3 的向量是 1024 维，差向量也是 1024 维。模板注释里记的是
「直接上 1024 维 `|a-b|` 会过拟合（实测掉到 0.81）」，
所以先 `TruncatedSVD(n_components=32)` 压到 32 维再加进去。
**特征不是越多越好，高维差向量会把噪声一起带进来。**

**5）本地模型的三个环境坑（都写进模板注释了）。**
- 别设 `HF_ENDPOINT=hf-mirror.com`：那个站只做 308 跳转、不是真镜像，
  会让 `huggingface_hub` 的元数据校验失败。正确做法是 `HF_HUB_OFFLINE=1` 纯本地跑。
- Windows 证书库里有坏证书，Python 3.9 的 `ssl.py` 会抛 `SSLError`，
  所以要在 `import sentence_transformers` **之前**给
  `ssl.SSLContext.load_default_certs` 包一层 try/except 补丁。
- 别用 `.apply(model.encode)` 逐条编码，慢几十倍，要整批传 list：
  `m.encode(sentences, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)`。

**6）向量归一化之后，点积就是余弦。**
`normalize_embeddings=True` 让输出已经是单位长度，
所以 `(e1 * e2).sum(1)` 就够了，不需要再调 `util.cos_sim` 循环一遍。
这和上面 `cos_rows` 是同一个道理：**先看向量有没有归一化，再决定怎么算相似度**。

**7）仓库目录表里这道题被记成了「回归」。**
`README.md` 的目录表里 08 写的是「回归」，但代码里 `Similar` 是 0/1、
模型是 `LogisticRegression`、评分是正类 F1，方法论文档的题型速查表也写的是「预测=1 的 F1」。
**数据和代码冲突时以代码为准** —— 我把这篇按二分类写了。

**8）这道题没有 `code/` 目录。**
我一开始还去 `code/08-text-similarity` 找 notebook，结果那里什么都没有，
所以这篇里所有「实测数字」都只能引用模板注释，我给不出自己的实验记录。

## 六、小结

这道题给我留下的最有用的一条习惯，是**先写基线再写模型**：

- 目标 0/1 且评分只算正类 F1 时，「全预测 1」是一个很难打但也很容易算的对手。
- 打不过它的时候，正确的动作是**回头查特征**，不是换模型、不是调参。
- 手写特征清单本身也是一次自查：如果 `cos_word` 和 `cos_char` 的相关系数接近 1，
  那就等于只写了一个。

如果再跑一次，我会加两件事：把「数字冲突」单独做成一个强规则特征，
以及切一小份验证集，老老实实量一遍「离线档 vs 满血档」到底差多少。
