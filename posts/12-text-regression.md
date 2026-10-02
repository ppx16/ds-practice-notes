# 12 文本回归

> 同一道题我写了两遍——第一遍只用学生写的那段摘要，第二遍把原文和题目也拉进来、再叠上语义相似度特征。**两条路线的模型是同一个 LightGBM，差别全在喂进去的东西上。**

| | |
|---|---|
| 题型 | 回归 |
| 数据 | 训练 5,732 行 × 4 列，测试 1,433 行 × 3 列；另有 4 行的文章表 |
| 提交 | `考生姓名_results.csv`，列：`student_id`, `article_id`, `score` |
| 评分 | MSE |
| 代码 | [`code/12-text-regression`](../code/12-text-regression) · [`templates/12_bge嵌入测试.py`](../templates/12_bge嵌入测试.py) |

## 一、题目在问什么

场景是自动批改：老师给一篇文章出个摘要任务，学生写一段摘要，老师给一个分数。
要建一个模型，看着学生的摘要把那个分数预测出来。

分数是**浮点数，而且有正有负**——不是 0~1 的百分比，也不是 1~5 的星级。
我拿到的训练数据里，`score` 最小是 -1.547，最大是 1.690。这个范围决定了后续所有事情：
不能用分类的框架，损失函数要按回归选，预测值也不能 clip 到 0 以上。

测试集里只有 `student_id`、`article_id`、`summary_text`，没有分数。评测按 MSE 排序，越小越好
（所以离群点会被重罚）。数据里还额外给了一张文章表，能把 `article_id` 关联回原文标题、
原文内容和老师出的摘要题目。**这张表是不同路线的分水岭**——第一版没用它，第二版全靠它。

## 二、数据长什么样

`training.xlsx`（5,732 行 × 4 列）：

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `student_id` | object | 5,732 | 学生编号（哈希串），唯一 |
| `article_id` | object | **4** | 文章编号，只有 4 个取值 |
| `summary_text` | object | 5,732 | 学生写的摘要，英文，长度从几十到几百词 |
| `score` | float64 | 1,015 | **目标列**，有正有负 |

`test.xlsx`（1,433 行 × 3 列）：同样三个字段，去掉 `score`。

`all_articles.xlsx`（4 行 × 4 列）：

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `article_id` | object | 4 | 主键，和上面两张表关联 |
| `summary_question` | object | 4 | 老师要求摘要的方向 |
| `article_title` | object | 4 | 原文标题 |
| `article_text` | object | 4 | 原文全文 |

几个一眼就能记住的事实：

- **只有 4 篇文章**，5,732 条训练摘要平均每篇 1,400 多条。所以 `article_id`
  是个 4 值类别而不是「高基数 ID」，做 one-hot 完全没问题。
- `score` 有 1,015 个不同取值 / 5,732 行，说明分数是很细的连续值，不是离散档位。
- 摘要里有**空字符串**（非空计数是满的，但内容可能是空的），这类样本的 `score` 是 0。

## 三、思路：两条路线

`code/12-text-regression/code/` 下面有两个目录，这是这次练习最值得看的部分：

```text
code/12-text-regression/code/
├── 1_Traditional/
│   └── LightGBM.ipynb        路线 A：只有摘要文本，一个 notebook 从头跑到尾
└── 2_LLM/
    ├── feature_construct.py  路线 B：特征工程，落盘成 train_features.csv / test_features.csv
    └── modeling.py           路线 B：读特征 → 多模型 → 集成 → 出提交文件
```

### 两条路线的对照

| | 路线 A（`1_Traditional`） | 路线 B（`2_LLM`） |
|---|---|---|
| 用到的字段 | **只有 `summary_text`** | `summary_text` + `summary_question` + `article_title` + `article_text` |
| 文本特征 | TF-IDF 5,000 维（1-2 gram） | TF-IDF 只用 3,000 维、只拿来**算相似度**，不进模型 |
| 语义特征 | 无 | BERT 句向量（`bert-base-uncased`）余弦相似度 ×2 |
| 人工特征 | textstat 的 10 个可读性指标 + 6 个长度/比例指标 | 长度比例、语言质量（重复词/句首大小写/超长句/括号不配对/TTR）、衔接词比例 |
| 文章层面 | `article_id` 做 one-hot | `article_id` 的**目标编码**（该文章平均分、标准差、平均词长） |
| 模型 | 单个 LightGBM | LightGBM + 随机森林对照 + 一个两层 MLP，最后简单平均集成 |
| 验证 | 5 折 KFold，用 OOF 的 RMSE 反推最终 `n_estimators` | 5 折 CV 比 MSE → `RandomizedSearchCV` 调参 → 神经网络 |
| 产物 | 一个 notebook + `LightGBM.pkl` | 两个脚本 + `train_features.csv` / `test_features.csv` 中间产物 |

**核心观察：两边的模型是同一个 LightGBM，参数风格也几乎一样**（`objective='regression_l1'`、
`num_leaves=31`、`colsample_bytree=0.8`、`subsample=0.8`、`reg_alpha/reg_lambda=0.1`）。
路线 B 多出来的全部东西都在**特征**这一侧——这验证了我一直信的判断：
**文本题里换模型收益很小，加工输入才是主战场。**

路线 B 的代价也很清楚：多了一个 BERT 前向、多了一层 n-gram 语言模型、多了一次全量
`iterrows` 遍历，还拆成两个脚本要维护中间 CSV。**它不是「更好」，是「更重」。**

## 四、关键代码

### 4.1 路线 A：纯文本也能扛

```python
def extract_features(text):
    words = text.split()
    sentences = [s for s in re.split(r'[.?!]+', text) if len(s) > 0]
    num_sentences = len(sentences) if len(sentences) > 0 else 1
    num_unique_words = len(set(w.lower() for w in words))
    features = {
        'num_chars': len(text), 'num_words': len(words),
        'num_sentences': num_sentences,
        'avg_word_length': len(text) / len(words) if words else 0,
        'avg_sentence_length': len(words) / num_sentences,
        'unique_word_ratio': num_unique_words / len(words) if words else 0,
    }
    try:
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['gunning_fog']         = textstat.gunning_fog(text)
        # ... smog / ARI / coleman_liau / linsear_write / dale_chall / syllable_count ...
    except Exception:
        for key in [...]:          # 文本太短时 textstat 会抛异常
            features[key] = 0
    return features
```

两点值得说：

- 这一层全是**能被人读懂的特征**：词数、句数、词汇丰富度、可读性等级。
  `textstat` 那几个指标本来是为「这段文字相当于几年级水平」设计的，
  拿来当「这段摘要写得好不好」的代理非常自然。
- 那个 `try / except` 不是装饰：**空摘要送进 `textstat` 会直接抛异常**，
  而这个数据集里确实有空摘要。没有兜底，整条流程会在循环里挂掉。

### 4.2 路线 A：TF-IDF 的 fit / transform 纪律

```python
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_train['joined_text'])   # 训练集：fit_transform

# ... 到测试集时：
tfidf_matrix_new = tfidf_vectorizer.transform(df_new['joined_text'])     # 只用 transform，绝不重新 fit
```

notebook 在测试集那一格专门写了一行注释提醒自己不要重新 fit。
**重新 fit 会让测试集的列名和含义跟训练集完全错位**，模型照样能 `predict` 出数，
但每个数都是垃圾——又一个「不报错但全错」的坑。内存紧的机器上这一步要格外小心。

顺带一个真实的代价：`tfidf_matrix.toarray()` 把稀疏矩阵转成稠密 DataFrame，
5,732 行 × 5,000 列 × 8 字节粗算就是 **约 220 MB**，后面 `pd.concat` 还会再复制一份。

### 4.3 路线 A：用交叉验证反推树的数量

```python
for fold, (train_index, val_index) in enumerate(kf.split(X, Y)):
    model = lgbm.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], eval_metric='rmse',
                     callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    optimal_rounds.append(model.best_iteration_)
    rmse_scores.append(np.sqrt(mean_squared_error(Y_val, model.predict(X_val))))

final_n_estimators = int(np.mean(optimal_rounds) * 1.1)   # 留 10% 余量
```

流程是：**5 折里每折都靠早停找到自己的最佳轮数 → 取平均 → 乘 1.1 → 用这个数在全量数据上重训一次。**
乘 1.1 的逻辑我认同：折内的训练集只有 80%，全量训练时数据变多了，
最优轮数通常会比折内多一点点。

notebook 里存下来的输出只有前两折（`RMSE 0.4473 / 0.4580`，最佳迭代次数 3365 / 3370），
后面几折的输出没有留在文件里，所以这里给不出完整的 5 折平均。
但这两折的数字已经能看出结论：**单靠摘要文本 + TF-IDF，RMSE 在 0.45 附近。**

### 4.4 路线 B：把原文和题目换成两个相似度

路线 B 最实质的升级在这里——它不只「多用了字段」，而是把字段**变成了可比的距离**：

```python
def bert_embedding(text):
    inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                            truncation=True, max_length=512)
    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = self.model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()   # 均值池化

emb_summary  = bert_embedding(summary_text)
emb_article  = bert_embedding(article_text)
emb_question = bert_embedding(summary_question)

return {
    "tfidf_sim_article":  cosine_similarity(tfidf_summary, tfidf_article)[0][0],
    "tfidf_sim_question": cosine_similarity(tfidf_summary, tfidf_question)[0][0],
    "bert_sim_article":   cosine_similarity(emb_summary, emb_article)[0][0],
    "bert_sim_question":  cosine_similarity(emb_summary, emb_question)[0][0],
}
```

值得记的是**同一个相似度用两种方法各算一遍**：TF-IDF 余弦抓的是「用词重叠」，
BERT 余弦抓的是「意思接近」。学生把原文换了说法复述时，前者会掉到 0、后者还在。
两组都留着，让 LightGBM 自己决定信谁——这是很省事也很对的思路。

`article_level_features` 那一组则是另一种套路：把 `article_id` 的**目标统计**
（该文章的平均分、分数标准差、原文平均词长）作为特征。4 篇文章、每篇 1,400 条样本，
这个统计量很稳，基本上等于告诉模型「这篇文章的分数大致在什么区间」。

### 4.5 路线 B：一个小 MLP + 简单平均集成

`modeling.py` 里除了 LightGBM 和随机森林，还有一个两层全连接：

```python
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1))

# Adam(lr=0.001) + MSELoss，训练 30~50 轮
```

最后是：

```python
if lgb_success and nn_success:
    models = [lgb_model_final, rf_model_final]
    final_pred = ensemble_predictions(models, X_test)   # np.mean(axis=0)
```

**注意这里有个逻辑对不齐**：判断条件是「LGBM 和 NN 都成功」，但集成用的是
`[LightGBM, 随机森林]`，NN 的 `y_pred_nn` 算了却没进集成。
结果就是——**只有当神经网络训练失败时，集成的分支才不会被走到**，也就是说
那个 `nn_success` 开关实际上控制的是「要不要集成两个树模型」。
这个 bug 不影响交付（两种分支产出的都是合法预测），但读代码的时候要能看出来，
不能以为是「三模型集成」。我把它当反面教材记下来了：**分支条件和分支内容必须一一对应**，
否则半年后自己都读不懂。

### 4.6 模板里的第三条路：换掉那个 BERT

`templates/12_bge嵌入测试.py` 是我后来补的对比实验，思路是「把第二版里那个 BERT 换成
本地已经下好的 `bge-m3`」，并且**把 A/B 对比直接写进一个脚本**：

```python
X_base      = hstack([csr_matrix(feat_train.values),  X_bow_train]).tocsr()          # 基线
X_full      = hstack([X_base, csr_matrix(B_tr)]).tocsr()                            # 加 bge 相似度

def run_cv(X, name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = LGBMRegressor(objective='regression_l1', metric='rmse', n_estimators=1500,
                          learning_rate=0.05, num_leaves=31, ...)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va])
    return mean_squared_error(y, oof)

mse_base = run_cv(X_base, 'TF-IDF 基线')
mse_bge  = run_cv(X_full, 'TF-IDF + bge-m3')
```

三个设计比原版好：**手工特征也用稀疏矩阵装**（`csr_matrix(...)` + `hstack`，
路线 A 是全 dense 拼接）；**bge 的 embedding 按 `article_id` 缓存**（只有 4 篇文章，
算一次存起来，省掉几千次重复编码）；**A/B 跑同一套 KFold**——比较两个特征方案必须固定折的划分，
否则差异里混进了抽样噪声。

一个细节：`run_cv` 返回的是 `mean_squared_error(y, oof)`，**这是 MSE 不是 RMSE**
（参数里的 `name` 只是打印标签）。题目按 MSE 排序，所以口径一致；
但别被 `metric='rmse'` 骗了——那是 LightGBM **训练时**的早停指标，跟这个数字不是一回事。

## 五、踩的坑

1. **目标编码把答案漏进了特征。** `feature_construct.py` 里
   `transform(df, is_train=True)` 对每一行调用
   `article_level_features(row, train_data=df)`，而 `df` 是**整个训练集**。
   也就是说 `article_avg_score` 里包含了这一行自己的 `score`。
   每篇文章 1,400 多条样本，单行的权重只有 1/1400，影响被稀释得很厉害，
   不是那种「线下 0.9 线上 0.4」的灾难性泄漏；但严格说这是定义上的泄漏，
   而且 `modeling.py` 的交叉验证是在**已经算好的特征**上做的，
   所以那份 CV MSE 是偏乐观的。正确写法是把这层统计放进折内算。
2. **dense 化 TF-IDF。** `tfidf_matrix.toarray()` 在 5,732 × 5,000 的规模下就是 200 MB 级别。
   模板版本改成了稀疏 `hstack`，这是实打实的改进。
3. **`get_dummies` 分开做。** notebook 先对训练集 one-hot、再对测试集 one-hot，
   然后自己检查了一遍列是否一致（`if list(X.columns) != list(X_new.columns): print("警告…")`）。
   这次只有 4 篇文章、两边都有所以没事；但只要某个类别只在测试集出现，
   测试集就会多一列、训练集少一列。**这个检查应该每个项目都写**，
   因为它把「静默错位」变成了一条警告。
4. **空摘要。** 数据集里有内容为空的摘要（分数为 0）。
   特征工程里必须处理：`textstat` 会抛异常（已用 try 兜住），
   分词会得到空列表（已用 `if words else 0` 兜住）。少任何一个都会在循环里炸。
5. **分数有正有负，别 clip。** 一开始我有种冲动想把预测值裁到 0 以上，
   但真实分数最小是 -1.547。裁掉负值等于系统性偏移，MSE 只会更差。
6. **两份数据、两种格式。** 路线 A 读的是 `training.csv` / `test.csv`，
   路线 B 读的是 `training.xlsx` / `test.xlsx`。同一批数据的两份拷贝。
   自己练的时候无所谓，但排查「为什么两条路线结果对不上」时，
   第一件要确认的事就是**它们读的到底是不是同一条数据**。

## 六、小结

- **这题最大的收获是那次 A/B。** 同样一个 LightGBM，加了原文相关性、语义相似度和
  文章级统计之后，输入侧多了十几列有意义的信息；模型一行没改。
  以后再遇到「文本回归效果一般」，我第一个反应是去挖特征，而不是去换模型。
- **文本特征有三层，可以逐层加**：
  ① 长度 / 比例 / 可读性这类**表面统计**（便宜、必做）；
  ② TF-IDF 这类**字面重叠**（便宜、必做）；
  ③ BERT / bge 这类**语义相似度**（贵，但能补上「换了说法」的情况）。
  路线 A 做到 ②，路线 B 做到 ③。
- **目标编码必须折内做。** 这题只是轻微稀释，但换个「每类只有几条」的场景就是灾难。
  这条纪律写死在心里比记住任何一个模型参数都值钱。
- **中间产物落盘是双刃剑。** `train_features.csv` / `test_features.csv` 让特征工程和建模解耦、
  可以分开调试；代价是 CV 只能在算好的特征上做，折内特征（比如目标编码）就没有立足之地了。
- 评测口径先看清是 MSE 还是 RMSE：这题是 MSE，**同一份预测两种口径排出来的名次可能不一样**。
