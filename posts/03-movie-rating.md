# 03 用户电影评分预测

> 表格里有两大列英文文本，看着像文本回归，其实是协同过滤；评分用 RMSE，但输出必须是 1~5 的整数，写小数直接 0 分。

| | |
|---|---|
| 题型 | 回归（评分预测 / 协同过滤） |
| 数据 | 训练 20,000 行 × 5 列（本地这份是子集；notebook 里跑的是 376,353 行），测试 20,000 行（原尺度 94,089 行） |
| 提交 | `<姓名>_results.csv`，列：`user_id`, `video_id`, `predicted_rank` |
| 评分 | RMSE |
| 代码 | [`code/03-movie-rating`](../code/03-movie-rating) · [`templates/04_回归_改进版.py`](../templates/04_回归_改进版.py)（本题没有专用模板，回归骨架借用它） |

## 一、题目在问什么

给一批「谁看了哪部片、打了多少分」的记录，再附上电影的英文名和一段剧情简介，
让我建一个模型去预测某个用户对某部电影的评分。评分是 1~5，1 表示完全没兴趣、5 表示很有兴趣，
目标字段就叫 `Rank`。

提交文件三列：`user_id`、`video_id`、`predicted_rank`。
其中预测分**必须是 1~5 之间的整数**，题目专门举了反例说 4.2 这种实数不行，
格式不对就算零分 —— 这一条和模型好坏没关系，是硬约束。

评分方式是按 RMSE 排名，越小越好。

我一开始被 `name_en` / `description_en` 这两列带偏了，以为要做文本挖掘，
后来才想明白：这两列是**电影级属性**，不是用户级信号。

## 二、数据长什么样

`training.xlsx`（本地这份子集，20000 行 × 5 列）：

| 列名 | 类型 | 非空 | unique | 说明 |
|---|---|---|---|---|
| `user_id` | int64 | 20000 | 12736 | 用户序号 |
| `video_id` | int64 | 20000 | 75 | 电影序号 |
| `name_en` | object | 20000 | 75 | 电影英文名 |
| `description_en` | object | 20000 | 75 | 剧情描述（很长的一段英文） |
| `Rank` | int64 | 20000 | 5 | **目标**：用户评分 1~5 |

`test-Ans.xlsx`（练习用的测试集，20000 行 × 5 列，字段同上，**带 `Rank`**，所以我能在本地直接算分）：

| 列名 | 类型 | 非空 | unique | 说明 |
|---|---|---|---|---|
| `user_id` | int64 | 20000 | 13444 | 用户序号 |
| `video_id` | int64 | 20000 | 222 | 电影序号 |
| `name_en` | object | 20000 | 221 | 电影英文名 |
| `description_en` | object | 20000 | 216 | 剧情描述 |
| `Rank` | int64 | 20000 | 5 | 练习集里的真实评分，用来算 RMSE |

两个数字值得停下来看一眼：

1. 20000 行训练数据里只有 **75 部电影**，也就是同一段 `description_en` 被复制了几百遍。
   文本列的信息量被行数严重放大了。
2. 测试集有 **222 部**电影，比训练集的 75 部还多。也就是说这份子集里存在**训练集没见过的电影**，
   电影偏差会取不到值，必须想好退路（填 0 等于退回全局均值）。

全量数据的规模我在 notebook 的输出里也看到了：测试集 94089 行，
`video_id` 分成 1004 组、`user_id` 分成 76107 组 —— 用户极度稀疏。

## 三、思路

### 1. 先判题型：是回归，不是 5 分类

`Rank` 是 1~5 的有序整数，评分是 RMSE。这两个条件加起来指向**回归**：
直接拟合一个连续值，让平方误差最小。

反过来，如果套 5 分类（`LGBMClassifier` 输出类别）会丢两样东西：
一是类别之间的序（3 分和 5 分的距离比 3 分和 4 分远），二是概率里蕴含的「不确定」，
最后 RMSE 反而更差。我在 `solve_2206.py` 的注释里也留了这条：
早先一版用分类器直接输出类别是错的。

### 2. 找信号源：文本是废特征，用户和电影才是信号

模型要回答的是「**这个用户**会不会喜欢**这部片**」。而 `name_en` / `description_en`
只跟电影有关，跟用户无关 —— 同一部电影的评论文本对所有用户是一模一样的。

这条判断有数据支撑：全量训练集 376353 行里 `video_id` 只有 1004 个、
`description_en` 只有 803 个不同值，文本被重复了几十万次；
笔记里还记了另一条 —— 全量数据里测试集出现的电影，训练集全都见过。
既然「哪部电影」这件事训练集已经见过，
那**用该电影的历史平均分就够了**，不需要读剧情简介。
（本地那份 2 万行子集不是这样，见「踩的坑」第 3 条。）

所以本题本质是**协同过滤式的评分预测**：全局均值 + 电影偏差 + 用户偏差。

### 3. 两个版本的基准预测器

- `Movie_Score.ipynb`：电影均分和用户均分取**几何平均** `round(sqrt(VRank * URank))`。
- `solve_2206.py`：全局均值 + 收缩偏差 `pred = μ + b_i + b_u`。
  偏差用 `sum / (count + λ)` 的形式收缩，防止只有一两条评分的电影/用户把方差炸开。

### 4. 最后一步：round 成整数

先算连续预测值、把 RMSE 调好，最后统一 `round` 并 `clip` 到 1~5。

## 四、关键代码

**① 电影均分 + 用户均分（参考 notebook 的几何平均版）**

```python
train = train.drop(columns=['name_en', 'description_en'])   # 文本列直接扔掉

vid_avg_rank = train[['video_id', 'Rank']].groupby('video_id').mean().reset_index()
vid_avg_rank['VRank'] = vid_avg_rank.pop('Rank')
uid_avg_rank = train[['user_id', 'Rank']].groupby('user_id').mean().reset_index()
uid_avg_rank['URank'] = uid_avg_rank.pop('Rank')

test = test.merge(vid_avg_rank, how='left').merge(uid_avg_rank, how='left')
test['URank'] = test['URank'].fillna(2.791864)   # 冷启动用户 → 全局用户均分
```

`fillna(2.791864)` 这个数不是拍的，就是 `uid_avg_rank.mean()` 的输出（notebook 里能看到）。
`how='left'` 决定了没见过的用户/电影会留下 NaN，这里必须先填掉再往下走。

**② 几何平均 + 取整**

```python
for i in range(test.shape[0]):
    test['Predicted_Rank'][i] = round(sqrt(test['VRank'][i] * test['URank'][i]))
```

几何平均是「电影好 + 用户宽容」两个因素的折中：电影均分低但用户手松，结果会往中间拉。
这个写法的副作用是 `SettingWithCopyWarning`（逐行赋值到切片），
老实说这是参考 notebook 里的坏习惯，换成列表推导或 `.apply` 就没这个问题。

**③ 偏差收缩版（`solve_2206.py`，我更认可的一版）**

```python
def fit_baseline(lam_i, lam_u, n_iter=6):
    b_i = pd.Series(0.0, index=item_cnt.index)
    b_u = pd.Series(0.0, index=user_cnt.index)
    for _ in range(n_iter):
        resid = r - mu - tr['user_id'].map(b_u).fillna(0).values
        g = pd.DataFrame({'v': tr['video_id'].values, 'x': resid.values}).groupby('v')['x'].agg(['sum', 'count'])
        b_i = g['sum'] / (g['count'] + lam_i)
        resid = r - mu - tr['video_id'].map(b_i).fillna(0).values
        g = pd.DataFrame({'u': tr['user_id'].values, 'x': resid.values}).groupby('u')['x'].agg(['sum', 'count'])
        b_u = g['sum'] / (g['count'] + lam_u)
    return b_i, b_u
```

交替地估电影偏差和用户偏差，各迭代 6 轮。关键在 `g['count'] + lam` 这个分母：
只有 1 条评分的电影，如果直接取残差均值就是「1 个人说了算」，
除以 `1 + λ` 之后偏离会被压到 1/(1+λ)，算是很朴素的收缩。

预测和取整：

```python
pred = mu + te['video_id'].map(b_i).fillna(0).values + te['user_id'].map(b_u).fillna(0).values
cont = np.clip(pred, 1.0, 5.0)                    # 连续预测，用于选 λ 和看 RMSE
pred_int = np.clip(np.round(cont), 1, 5).astype(int)   # 提交用这一列
```

`.fillna(0)` 在这里是「不知道该加多少偏差，就不加」，结果自然退回全局均值 `μ`；
`clip(1, 5)` 是防止 λ 选小了之后预测跑到 0.7 或 5.4 这种区间外。

**④ 参考 notebook 里那个文本路线（`code/reference/`）**

那一版走的是另一条路：nltk 分词 + 词性还原 + 去停用词，拼上 `name_en` 和 `description_en`，
再做 TF-IDF / CountVectorizer，接 SVR、LogisticRegression、BernoulliNB，打印 MSE。

```python
train_df['text'] = train_df['name_en'] + train_df['description_en']
train_df['text'] = train_df['text'].apply(lambda x: preprocessing_text(x, stop_words))
word_mtx = vec_methods[vec_num].transform(train_df['text'])
svm_clf = SVR().fit(word_mtx_de, train_df['Rank'])
print(mean_squared_error(Y_test, Y_pred))
```

**这几个 cell 的输出没有留在 notebook 里，所以我不知道它当时跑到多少。**
我能确定的是两件事：一是它只打印 MSE 没有落盘分数，二是它最后没被采用
（真正交出去的是纯 `video_id` / `user_id` 的协同过滤解），
这跟上面「文本是电影级属性」的判断是一致的。

## 五、踩的坑

**1）小数预测直接 0 分，取整是格式题不是模型题。**
题目要求 `predicted_rank` 是 1~5 的整数，所以哪怕连续预测的 RMSE 再低，
只要交出 4.2 就是格式错误。取整确实会掉一点分：笔记里记的是连续 RMSE 1.393，
round 之后变成 1.424。**先把连续值调到最好，最后一步再取整**，
不要在中间环节就把预测锁成整数（那样梯度/残差都变粗糙了）。

**2）带答案的测试集能用来挑超参，但正式跑要换成交叉验证。**
`test-Ans.xlsx` 里有 `Rank`，所以 `solve_2206.py` 是直接在测试集上网格搜 `λ` 的
（`lam_i ∈ {0.5,1,2}`、`lam_u ∈ {2,3,5}`）。脚本注释里也写明了这只是练习期方便，
真正提交时应该对训练集做 K 折、用 OOF RMSE 选 λ。

**3）冷启动没兜住会直接崩。**
本地这份子集里测试集有 222 部电影、训练集只有 75 部，
`map` 之后一定出现 NaN。几何平均那版只填了 `URank`，
如果 `VRank` 缺失，`sqrt(NaN)` 会污染整列 —— 我现在会两个都填。

**4）参考 notebook 的输出只截了上界。**
`test.ipynb` 里是这么处理的：

```python
if Y_pred[i] > 5:
    Y_pred[i] = 5
temp.append('%.0f' % Y_pred[i])
```

只压了 `> 5`，没有托底。SVR 预测出 0.4 甚至负数都很正常，
`round` 之后就是 0 或负整数，直接违反「必须 1~5」。
正确写法是两边都夹：`np.clip(np.round(pred), 1, 5)`。

**5）别被文本列骗去逐行编码。**
我确实动过「用 BGE 给每行文本算 embedding」的念头，幸好算了一下量级：
全量训练 + 测试是 37.6 万 + 9.4 万 ≈ 47 万行，
而这里面只有约 1600 条不同的文本。既慢（几十分钟起）又对本题毫无帮助 ——
`solve_2206.py` 的注释把这条记成了「慢的根源」。

**6）`video_id` 别当数值特征喂进模型。**
它只是个编号，大小没有含义。这题的解法里它只用来 `groupby` 和 `map` 偏差，
没有进过特征矩阵。

## 六、小结

这道题教会我的事，主要不是「怎么写协同过滤」，而是**先看 unique 数再决定特征怎么用**：

- `description_en` 有 20000 行但只有 216~803 个不同值 → 它不是文本特征，是电影 ID 的另一种写法。
- 既然信号在「谁 × 哪部片」，就先建 `μ + b_item + b_user` 的基准，再谈别的。
- RMSE 是连续指标，但交付格式可能是离散的 —— **指标和格式是两件事，读题时要分开记**。

如果重做一遍，我的顺序会是：
先跑「全填全局均值」拿到保底 RMSE，再上偏差收缩，
最后再考虑要不要给用户/电影加隐向量（这题数据量下大概没必要）。
