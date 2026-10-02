# 06 相似项检索（同款商品贴文）

> 输出是一串 ID 列表而不是类别，评分是 Macro-F1 —— 这是检索题不是分类题；
> 图片的感知哈希看着最像「图像特征」，实际几乎没用，真正起作用的是标题。

| | |
|---|---|
| 题型 | 检索 / 排序（每笔输出 30~50 个 ID） |
| 数据 | 训练 34,230 行 × 3 列（本地卡片记的是 20,000 行那份），测试 20 行 |
| 提交 | `<姓名>_results.csv`，列：`Posting_ID`, `Num`, `Predicted_Results` |
| 评分 | Macro-F1 |
| 代码 | [`code/06-similar-item`](../code/06-similar-item) · [`templates/06_相似项检索.py`](../templates/06_相似项检索.py) |

## 一、题目在问什么

同一件商品会在不同渠道、被不同卖家重复上架，照片和标题写法都不一样。
题目给一批商品贴文，每笔有贴文 ID、图片信息、标题描述，让我判断**哪些贴文其实是同一件商品**。

图片文件太大不方便给，所以图片只给了一个 **Phash（感知哈希）**，16 个十六进制字符，
可以理解成图片的指纹：哈希越接近，两张图越像。题目还提示了算法 ——
把 16 个十六进制字符转成 64 位二进制，然后数两个二进制串有多少位不同（汉明距离）。

我拿到的测试集只有 **20 笔**。对每一笔，要去训练集里找出同一商品的其它贴文，
输出 30~50 个贴文 ID，用英文逗号分隔。输出几条由我自己定，不同的测试样本可以条数不同。

提交三列：`Posting_ID`、`Num`（条数，必须在 30~50 之间）、`Predicted_Results`（逗号分隔的 ID）。
评分是 Macro-F1。**`Num` 超出 30~50 直接归零**，这条是硬格式约束。

## 二、数据长什么样

`train.csv`（本地这份 20000 行 × 3 列）：

| 列名 | 类型 | 非空 | unique | 说明 |
|---|---|---|---|---|
| `Posting_ID` | int64 | 20000 | 20000 | 贴文主键，也是输出要填的东西 |
| `Posting_Image_Phash` | object | 20000 | 17720 | 图片感知哈希，16 位十六进制字符串 |
| `Posting_Title` | object | 20000 | 19610 | 商品标题，多为印尼语/英文混排 |

`train_ans.csv`（20000 行 × 4 列）在同样三个字段上多一列：

| 列名 | 类型 | 非空 | unique | 说明 |
|---|---|---|---|---|
| `label_group` | int64 | 20000 | 6416 | 同一件商品共享同一个分组编号（练习用的答案） |

`test_ans.csv`（20 行 × 5 列，练习测试集，带答案分组）：

| 列名 | 类型 | 非空 | unique | 说明 |
|---|---|---|---|---|
| `Posting_ID` | int64 | 20 | 20 | 查询贴文 |
| `Posting_Image_Phash` | object | 20 | 20 | 图片哈希 |
| `Posting_Title` | object | 20 | 20 | 标题 |
| `Number_of_the_Same_Page` | int64 | 20 | 10 | 同款商品总数（含自己），如 45 / 41 / 37 |
| `label_group` | int64 | 20 | 20 | 该商品的答案分组编号 |

`test_results.csv`（20 行 × 5 列）是配套的参考结果，直接给了答案 ID 列表：

| 列名 | 类型 | 非空 | unique | 说明 |
|---|---|---|---|---|
| `Number_of_the_Same_Page` | int64 | 20 | 10 | 同款总数，如 45 |
| `Check_Number` | int64 | 20 | 10 | 比上面少 1，如 44 |
| `label_group` | int64 | 20 | 10 | 分组编号 |
| `ans` | object | 20 | 20 | 答案 ID 列表，逗号分隔 |

`Number_of_the_Same_Page` 比 `Check_Number` 大 1 很好理解：
总数算了查询自己，而 `ans` 里列的是「其它」同款。
所以第一笔的真实同款数是 44 个，第四笔是 35 个 —— **每笔的答案长度都不一样**。

`ans.csv`（20 行 × 2 列）和 `练习2303_results.csv`（20 行 × 3 列）是我自己跑出来的中间产物，
分别是 `Posting_ID + ans_my` 和 `Posting_ID + Num + Predicted_Results`。

三个和卡片对不上的地方，我如实记一下：

1. 题目描述里的训练规模是 34,230 笔，notebook 里 `train_df` 丢掉 1 条非法记录后是 **34229 行**，
   但本地数据卡片记的是 `train.csv` 20000 行 —— 两个数字不是一份东西，我按各自来源分开看。
2. `Posting_Image_Phash` 有 20000 行但只有 17720 个唯一值，**同一个哈希重复出现**，
   说明有些贴文用了完全相同的图。
3. `train_ans.csv` 的 `label_group` 是 6416 组 / 20000 行，平均一组 3 条左右，
   而测试集第一笔的同款（`Number_of_the_Same_Page` 45、`Check_Number` 44）就有 44 条，
   两边完全不是一个量级 —— 训练集的分组规模不能拿来猜测试集该输出多少条。

## 三、思路

### 1. 先判题型：不是分类，是检索

输出是「一串 ID」而不是一个标签，评分是 Macro-F1（逐笔算 F1 再平均），
所以这是一道**相似项检索**题。这一点想错，后面全错 ——
套分类模型的思路（把 `label_group` 当类别去预测）根本没法产出 ID 列表。

### 2. 两条相似度路线，先各自取 Top-N 再合并

`ans.ipynb` 的做法很直白，两个信号各出一路：

- **图片路**：Phash → 64 位二进制 → 汉明距离，取最近的 25 条。
- **文本路**：标题过 `all-MiniLM-L6-v2` 拿句向量 → 余弦相似度，取最近的 25 条。
- **合并**：两路求并集，最多 50 条。

### 3. 但我对图片那一路的信任度很低，理由在后面「踩的坑」里

模板⑥里记的实测结论更狠：**Phash 对这道题没用**。
正确对和错误对算出来的汉明距离都在 32 附近，基本是随机水平 ——
不同卖家拍同一件商品角度不同，感知哈希聚不到一起。
真正的主信号是标题的字符级相似度。

模板里选的是 `char_wb(3,4)` 字符 n-gram TF-IDF 余弦（标题小写化后直接算，
印尼语标题不需要分词），再叠一层「2-hop 桥接」：
拿 Top-10 当种子，把「种子 → 其它贴文」相似度 ≥0.55 的候选也拉进来，与直接 Top-30 取并集。

### 4. 输出条数

题目允许 30~50，模板注释里记的实测是「F1 随 Num 单调升到 50（上限）」，所以尽量填满。
这是有算术原因的：评分是逐笔 F1，而每笔真实同款数大多在 34~44 之间，
输出条数越多、召回越容易上去；精度虽然会被摊薄，但在这个区间里召回涨得比精度掉得快。

## 四、关键代码

**① Phash → 二进制位**

```python
All_df['to_binary'] = All_df['Posting_Image_Phash'].apply(
    lambda x: [int(i) for i in (bin(int(x, 16))[2:])])
```

先 `int(x, 16)` 把十六进制串转整数，`bin()` 再转成 `0b...` 字符串，
`[2:]` 切掉前缀，最后逐字符转成 0/1 的列表。

**② 汉明距离，取最近 100 条**

```python
def distance(a_):
    tmp = np.array([a_] * train_df.shape[0])
    every_dis = pd.Series(np.sum(np.abs(tmp - to_binary_list), axis=1), index=train_df.index)
    return every_dis.sort_values()[:100].index.tolist()
```

`np.abs(a - b)` 逐位相减取绝对值再按行求和，正好就是「不同的位数」。
这就是题目提示里那个算法，用 numpy 广播一次算完 3 万多行。

**③ 标题句向量 + 余弦**

```python
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
All_df['embedding'] = All_df['Posting_Title'].progress_apply(lambda x: model.encode(x))

def cos_sim(e_):
    cosSim_lst = [util.pytorch_cos_sim(e_, embeddings[idx]) for idx in range(shape0)]
    every_sim = pd.Series(cosSim_lst, index=All_df_idx)
    return every_sim.sort_values(ascending=False)[:100].index.tolist()
```

**④ 两路取并集**

```python
ans['ans_my'] = ans.apply(
    lambda x: list(set(x['nearest_ids'][:25]).union(set(x['cosSim_id'][:25])) - set(x.index)),
    axis=1)
```

`nearest_ids` 和 `cosSim_id` 上一步各留了 100 条，这里各取前 25 再并集，上限 50。

**⑤ 本地算分（用 `test_results.csv` 当答案）**

```python
concat_result['recall']    = concat_result['cnt_intersection'] / concat_result['cnt_rstAns']
concat_result['precision'] = concat_result['cnt_intersection'] / concat_result['cnt_myAns']
concat_result['f1'] = (2.0 * concat_result['recall'] * concat_result['precision']
                       / (concat_result['recall'] + concat_result['precision']))
```

逐笔算 precision / recall / F1，最后 `.mean()` 平均 —— 这就是 Macro-F1 的定义。

**⑥ 模板⑥里的检索方案（我更愿意重跑的一版）**

```python
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2,
                        max_features=200000, sublinear_tf=True)
X = tfidf.fit_transform(train_title + test_title)
Xtr, Q = X[:len(train)], X[len(train):]
S = cosine_similarity(Q, Xtr)

seeds  = np.argsort(-sim)[:SEED_N]
bridge = cosine_similarity(Xtr[seeds], Xtr).max(axis=0)
```

`char_wb` 是「按词边界切、但统计字符 n-gram」，所以同一件商品换个说法、
拼写有小差异、多几个噪声词，都能留下重合的 3~4 字符片段。
`fit_transform(train_title + test_title)` 把测试标题也放进语料里学词表，
避免测试集出现的字符组合被 `min_df=2` 过滤掉。

## 五、踩的坑

**1）Phash 会被 pandas 读成浮点数。**
有一条记录的哈希是纯数字的十六进制串，pandas 直接当数值读，
在 notebook 里显示成 `9.79E+108`。这样一转二进制就完全错了。
模板⑥里的处理是读文件时就把这一列钉死成字符串：

```python
train = pd.read_csv(TRAIN_PATH, dtype={PHASH_COL: str})
```

**2）只校验了字符串长度，没校验二进制位宽。**
notebook 里用 `if len(row['Posting_Image_Phash']) != 16` 挑非法记录，
这一条确实抓到了上面那个被读成浮点的值（它的字符串只有 10 个字符），很好。
但 `bin()` 会**丢掉前导零**：只要有一条哈希以 `0` 开头，
`bin(int(x, 16))[2:]` 得到的就不是 64 位，后面两个不等长的列表做数组减法会直接报错。
稳妥写法是 `bin(int(x, 16))[2:].zfill(64)`。
这次没炸说明这批数据恰好都以 8~f 开头，但风险是真的。

**3）`- set(x.index)` 是个空操作。**

```python
list(set(x['nearest_ids'][:25]).union(set(x['cosSim_id'][:25])) - set(x.index))
```

`x` 是一行（`apply(axis=1)`），`x.index` 是**列名的 Index**，
`set(x.index)` 得到的是 `{'Posting_Image_Phash', 'Posting_Title', ...}` 这堆列名，
减掉它对 ID 列表毫无影响。我猜本意是「去掉自己」，但训练集 ID 和测试集 ID 本来就不重叠，
去掉自己也是多余的。

**4）输出条数直接顶着上限，精度有天花板。**
测试集第一笔的 `Check_Number` 是 44，`ans` 里最多只有 44 个正确答案，
我却输出 50 条 —— 就算全部命中，精度也只有 44/50 = 0.88，F1 到不了 1。
这不是 bug，是 `Num=50` 的代价，要认。

**5）两处记录的分数不是一回事，不能混着看。**

- `ans.ipynb` 里我自己算的：平均 **F1 = 0.4200**、**precision = 0.3601**、**recall = 0.5089**
  （这三个数就是 notebook 的输出）。
  它是 Phash Top-25 ∪ MiniLM Top-25，对照 `test_results.csv` 的答案算的。
- 模板⑥注释里记的另一组实验：`char_wb(3,4)` TF-IDF Top-35 约 **Macro-F1 0.692**，
  再叠 2-hop 桥接约 **0.753**（对应 precision≈0.70、recall≈0.83）。

两组数字方法不同，对照的答案是不是同一份我也不能确定，所以**别把它们放在一起比较**。
我能确定的是：文本路比图片路有效，而这正是模板⑥里那句「官方模板用 Phash+标题并集，
其实是靠标题那一路在起作用」。

**6）慢的不是检索，是逐条编码。**
我一直以为 3 万多条 × 64 位的汉明距离会很慢，结果看进度条：
`distance()` 那一步 20 笔只花了 **5 秒**（20/20、3.71 it/s），
因为 `np.abs(tmp - to_binary_list).sum(axis=1)` 是用 numpy 广播一次算完 3 万多行的。

真正慢的是标题编码那一步：`All_df['embedding'].progress_apply(lambda x: model.encode(x))`
逐条调用，**34249 条跑了 12 分 09 秒**（46.92 it/s）。
同样的模型改成整批传 list（`model.encode(titles, batch_size=64)`）能快一到两个数量级 ——
模板⑧里专门把这条记成了坑：「别用 `.apply(model.encode)` 逐条编码」。

**7）别用分类的思路去做。**
`train_ans.csv` 里有 `label_group`，很容易手滑去训练一个「预测分组编号」的分类器。
但测试集的 `label_group` 是 20 行 20 个唯一值（每笔一个独立编号），
预测出这样一个编号没有任何用处 —— 这题要的是**排序**，不是分类。
`label_group` 唯一的正当用法是：拿它当验证集的答案，量一下我的检索方案到底行不行。

## 六、小结

这题真正教我的两件事：

- **看「输出 + 评分」判题型**：输出一串 ID、按 F1 排名 → 检索题。
  数据里有 `label_group` 这种「像标签的东西」反而是干扰项。
- **图片特征不一定比文本特征强**。感知哈希听起来很高级，
  但同一商品不同卖家会换图、换角度，哈希值就散了；
  反倒是商品标题里的品牌名、型号、规格这些字符片段最稳。

以后再碰到检索题，我的顺序是：
先跑一个纯标题 TF-IDF 的 Top-K 当保底，再用本地答案（或者从 `label_group` 切一小部分当验证集）
量一下 MiniLM / BGE 这些向量模型有没有更好，最后才决定 `Num` 填多少。
