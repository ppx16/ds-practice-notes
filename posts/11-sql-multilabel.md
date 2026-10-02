# 11 SQL 取数 + 多标签

> 这题的表结构比模型难：一个问题的答案在另一张表里、一个 Id 对应好几行、标签用逗号挤在一格——**先把数据摊平，模型反而是最简单的一段**。

| | |
|---|---|
| 题型 | 文本 · 多标签 |
| 数据 | 问题表 62,618 行 × 6 列；答案表 63,260 行 × 6 列；标签表 76,240 行 × 2 列（50,094 个不同 Id）；测试 12,524 行 × 1 列 |
| 提交 | `考生姓名_results.csv`，列：`Id`, `Tag` |
| 评分 | Micro-F1 |
| 代码 | [`code/11-sql-multilabel`](../code/11-sql-multilabel) · [`templates/11_特殊数据加载.py`](../templates/11_特殊数据加载.py) · [`templates/11_多标签分类_实战.py`](../templates/11_多标签分类_实战.py) |

## 一、题目在问什么

数据来自一个技术问答网站（统计类问题的问答社区）。每个问题可以被打上**多个**标签，
像 `bayesian,probability` 这样。

要做的事分两半：

1. 问题的标题、正文在**一张表**里，答案在**另一张表**里，标签在**第三张表**里，
   测试集只给了 Id 列表。要按主键把它们拼起来。
2. 拼好之后，用「标题 + 正文 + 答案」这坨文本，预测每个问题该有哪些标签，
   多个标签用逗号连成一个字符串交上去。

评分按 Micro-F1 算。标签是稀疏的多标签问题，所以 F1 比准确率合理得多——
把所有标签都预测成最常见那一个，准确率也许不难看，Micro-F1 会当场揭穿。

## 二、数据长什么样

先说一句**关于下面这些表格的口径**：仓库里的字段卡片是我用脚本自动抽的，
那个脚本给 CSV 设了「只读前 20000 行」的上限，所以卡片上写的 `20000 行` 是**截断后的**。
真实的完整行数我是从 notebook 里那段 `SELECT COUNT(*)` 的打印结果读到的，
下面两个数（62,618 / 63,260 / 76,240）用的是后者。

### 标签表 `Train_data.csv`（76,240 行 × 2 列）

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `Id` | int64 | 50,094 | 问题编号。**一行 = 一个（问题, 标签）对**，所以行数远大于问题数 |
| `Tag` | object | 30 | 单个标签名，如 `bayesian`、`distributions` |

50,094 个问题摊成 76,240 行，平均每个问题约 1.5 个标签。
「一行一个标签」是长表结构，也是这题多标签的来源。

### 问题表 `Filtered_Questions.csv`（62,618 行 × 6 列）

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `Id` | int64 | 62,618 | 问题编号，主键 |
| `OwnerUserId` | float64 | 12,468（前 2 万行） | 提问者，有缺失 |
| `CreationDate` | object | 19,998（前 2 万行） | ISO 格式时间串 |
| `Score` | int64 | 100（前 2 万行） | 问题得分 |
| `Title` | object | 19,993（前 2 万行） | 问题标题 |
| `Body` | object | 19,999（前 2 万行） | 问题正文，**带 `<p>` `<a href>` 等 HTML 标签** |

### 答案表 `Filtered_Answers.csv`（63,260 行 × 6 列）

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `Id` | int64 | 63,260 | **答案自己的编号**（不是问题编号） |
| `OwnerUserId` | float64 | 2,550（前 2 万行） | 答题者，有缺失 |
| `CreationDate` | object | 19,995（前 2 万行） | 时间串 |
| `ParentId` | int64 | 11,380（前 2 万行） | **答案所属问题的编号** —— join 就靠这一列 |
| `Score` | int64 | 114（前 2 万行） | 答案得分 |
| `Body` | object | 19,999（前 2 万行） | 答案正文，同样带 HTML |

### 测试与提交

| 文件 | 形状 | 说明 |
|---|---|---|
| `Test_data.csv` | 12,524 行 × 1 列 | 只有 `Id` |
| `Test_data_Ans.csv` | 12,524 行 × 2 列 | `Id`, `Tag`（练习用的答案） |
| 提交文件 | 12,524 行 × 2 列 | `Id`, `Tag`，`Tag` 是逗号连起来的标签串 |

测试集答案里出现过 1,135 种不同的标签组合，我交出去的那版是 1,126 种——
量级对得上，说明没有退化成「每个问题只给一个标签」。

## 三、思路

我把这题拆成两个独立的问题：

**第一半：把四个 CSV / 一张 SQLite 库摊平成两张宽表。**

```text
Train_data(Id, Tag)  ──┐
                       ├─ LEFT JOIN on Id ──> Filtered_Questions(Id, Title, Body)
Test_data(Id)        ──┘                              │
                                                      ├─ LEFT JOIN
Filtered_Answers(ParentId, Body) ─────────────────────┘   on Id = ParentId
                                                      │
                                                      ▼
                                       按 Id 聚合（一个 Id 可能有多行）
                                                      │
                                                      ▼
                                 training / test  各一张宽表
```

关键点有三个：**join 的键是 `ParentId = Id`**（不是 `Id = Id`）、
**一个 Id 会 join 出多行**（一个问题有多条答案）、**标签必须在聚合时一起合并**
（先 `drop_duplicates` 会把多标签压成一个）。

**第二半：多标签分类。** 有 `Tag` 列是逗号分隔的字符串，所以直接接模板里的做法 B：
`MultiLabelBinarizer` 把标签串转成 0/1 矩阵，再套 `OneVsRestClassifier`。

`code/11-sql-multilabel/csv2db.ipynb` 这个 notebook 其实只干了「灌库」这一件事：
把四个 CSV 用 `to_sql` 写进一个 SQLite，再 `read_sql` 把每张表的字段数和行数打出来核对。
它不建模，但这个动作很值得留着——**它是数据形状的自检**，
后面所有 join 都建立在这份「我以为的表结构」上，先确认一次再往下走。

## 四、关键代码

### 4.1 灌库 + 核对字段

```python
import sqlite3, pandas as pd

conn = sqlite3.connect("alldata.db")
for name, f in [("Filtered_Questions", "Filtered_Questions.csv"),
                ("Filtered_Answers",   "Filtered_Answers.csv"),
                ("Test_data",          "Test_data.csv"),
                ("Train_data",         "Train_data.csv")]:
    pd.read_csv(f).to_sql(name, conn, if_exists="replace", index=False)

for table in ["Filtered_Questions", "Filtered_Answers", "Test_data", "Train_data"]:
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    print(f"表格 `{table}` 有 {len(df.columns)} 個欄位，{len(df)} 筆數據")
```

四行 `print` 打出来的是 `6 / 62618`、`6 / 63260`、`1 / 12524`、`2 / 76240`。
这一步花的几十秒很值：**没有它我不会知道 `Train_data` 有 7 万多行**——
字段卡片上写的是 20000（截断上限），照着卡片的心智模型去估标签密度会错得很离谱。

### 4.2 join 的键容易写错

```python
training = (train_tag.merge(questions, on='Id', how='left')
                    .merge(answers, left_on='Id', right_on='ParentId', how='left')
                    .drop(columns='ParentId'))
```

第二段 merge 写的是 `left_on='Id', right_on='ParentId'`。
`Filtered_Answers` 的 `Id` 是**答案自己的编号**，跟问题编号完全无关；
能和问题对上的是 `ParentId`。写成 `on='Id'` 不会报错，只会把答案全 join 丢，
然后你会得到一个「答案列全是 NaN 但流程跑得通」的结果——这种安静的错误最难查。

### 4.3 聚合：文本拼接 + 标签合并

```python
def _join(s, sep=' '):
    return sep.join(x for x in s.dropna().astype(str).unique() if x.strip())

def agg_by_id(df):
    agg = df.groupby('Id').agg(
        Title =('Title',  lambda s: _join(s)),
        Body  =('Body',   lambda s: _join(s)),
        Answer=('Answer', lambda s: _join(s)),
    ).reset_index()
    if 'Tag' in df.columns:                       # 训练集才有标签
        tags = df.groupby('Id')['Tag'].apply(
            lambda s: ','.join(sorted(set(s.dropna().astype(str))))).reset_index(name='Tag')
        agg = agg.merge(tags, on='Id')
    return agg
```

这段是整个练习里最该背下来的一段，因为它同时修了两个错：

- **文本列用空格拼**（同一问题的多条答案合成一段），
- **标签列用逗号拼**（`','.join`，因为 `MultiLabelBinarizer` 那边是按逗号 split 的）。

模板的上一版这里是**标签用空格 join**，而下游按逗号 split——最后一个多标签问题被当成
一个叫 `bayesian probability` 的超长标签，标签数直接对不上。模板注释里专门记了这条：
**上游拼接符和下游切分符必须是同一个字符**。

第二个修的是顺序：早期版本为了「去重」先 `drop_duplicates(['Id','Tag'])` 再聚合，
看着合理，实际上把同一个问题的多个标签行压成了一行，**多标签退化成单标签**。
正确做法是聚合的时候一起合并，而不是提前去重。

### 4.4 清 HTML

```python
def clean_html(s):
    s = re.sub(r'<[^>]+>', ' ', str(s))        # 去 <p> <a href> 等标签
    s = re.sub(r'&[a-z#0-9]+;', ' ', s)        # 去 &nbsp; &lt; 等实体
    s = re.sub(r'[^a-z0-9]+', ' ', s.lower())  # 非字母数字 -> 空格
    return re.sub(r'\s+', ' ', s).strip()

training['text'] = (training['Title'].astype(str) + ' '
                    + training['Body'].astype(str) + ' '
                    + training['Answer'].astype(str)).apply(clean_html)
```

正文和答案里全是 HTML，`<p>` 和 `&nbsp;` 会变成高频「词」混进 TF-IDF。
三行正则的顺序也有讲究：**先去标签，再去实体，最后才做「非字母数字 → 空格」**。
顺序颠倒的话 `&nbsp;` 会先被拆成 `nbsp` 留下来。

### 4.5 多标签：二值化 + OneVsRest + 阈值

```python
tags_list = [[t.strip() for t in ts.split(',') if t.strip()] for ts in training['Tag']]
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(tags_list)                 # 变成 (样本, 标签数) 的 0/1 矩阵

tf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=30000)
tf.fit(pd.concat([training['text'], test['text']]))   # train+test 一起 fit，保证词表一致
X_tr, X_te = tf.transform(training['text']), tf.transform(test['text'])

clf = OneVsRestClassifier(
    LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear'))
clf.fit(X_tr, y)
proba = clf.predict_proba(X_te)                  # 每一列 = P(该标签 = 1)

p = (proba >= thr[None, :]).astype(int)
for i in range(p.shape[0]):                      # 兜底：一个标签都没过阈值
    if p[i].sum() == 0:
        p[i, np.argmax(proba[i])] = 1

out = pd.DataFrame({'Id': test['Id'], 'Tag': [','.join(ls) for ls in mlb.inverse_transform(p)]})
```

几个决定：

- **`class_weight='balanced'` 而不是 SMOTE。** 标签很稀疏，本机又没装 imbalanced-learn，
  直接用类别权重对冲正负样本不平衡，简单且不会崩。
- **TF-IDF 在 train + test 上一起 fit。** 只是词表，不涉及标签，所以不算泄漏；
  分开 fit 会让两边词表不一致，测试集的特征维度对不上训练集。
- **阈值不是一个 0.5 走天下。** 实战模板里试了三种：固定 0.5、全局搜一个最优阈值、
  逐标签各搜一个阈值。逐标签搜对低频标签（正例少于 10 个）会过拟合，
  所以那些标签固定回 0.5。三种方案的 Micro-F1 都打印出来对比，谁高用谁。
- **空标签兜底。** `inverse_transform` 遇到全 0 的行会给出空 tuple，交上去就是空字符串。
  所以做了「所有标签都不过阈值时，取概率最大的那一个」的兜底——
  理由和 09 那题一样：**空值或者单一取值都是提交格式上的雷**。

## 五、踩的坑

1. **join 键写错不报错。** `Id = Id` 和 `Id = ParentId` 都能跑通，后者才对。
   这类「静默错误」只能靠 join 完立刻检查 `Answer` 列的缺失率来发现。
2. **先 drop_duplicates 再聚标签。** 多标签题最经典的自杀方式，标签静默丢一半。
3. **上游拼接符和下游切分符不一致。** 空格 join + 逗号 split，
   结果是把一整串当成了一个标签。两端必须约定同一个分隔符。
4. **CSV 编码。** 问题/答案表带 BOM，`pd.read_csv` 要用 `encoding='utf-8-sig'`；
   早期版本用过 `ISO-8859-1`，能读进来但会把非 ASCII 字符读成乱码。
5. **字段卡片上的行数上限。** 卡片里 CSV 一律 20000 行，那是抽取脚本的截断，
   不是真实规模。真实规模要靠 count 或者 `.shape` 自己确认——
   这次差点因为「以为训练集只有 2 万行」而低估了标签密度。
6. **一个 Id 多行的 join 会放大行数。** join 完 `training` 的行数远大于问题数，
   这是正常的；但如果在此之上再做一次 `merge` 而不聚合，行数会继续翻倍。
   检查手段就是每一步都 `print(df.shape)`。

## 六、小结

- 这题的难度**九成在数据加载**。四张表、两个 join 键、一对多的关系、
  逗号分隔的标签列——把这一坨理清楚之后，建模就是「TF-IDF + 逻辑回归」的常规操作。
- 我把这段加载逻辑单独抽成了模板 [`templates/11_特殊数据加载.py`](../templates/11_特殊数据加载.py)，
  因为**「一个主键对多行 + 一个格子塞多个标签」这个形状在真实数据里反复出现**，
  跟题目是什么领域的没关系。
- 多标签的评测指标如果是 Micro-F1，**阈值比模型重要**。
  逐标签搜阈值、全局搜阈值、固定 0.5 三选一，成本很低，值得每次都跑一遍对比。
- 最实用的一条纪律：**每做完一步 join 或聚合就 `print(shape)`**。
  这题所有的坑都是「不报错但结果错」，只有形状会说话。
