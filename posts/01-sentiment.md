# 01 影评情感分析

> 拿到一段英文商品评论，判断它到底是好评还是差评 —— 这次学到的是：先跑通全流程再谈优化，以及本地评估一旦用错列，分数全是假的。

| | |
|---|---|
| 题型 | 文本 · 二分类 |
| 数据 | 训练 18789 行 × 10 列，测试 4697 行 × 10 列 |
| 提交 | `Final_Results.csv`，列：`CID`, `Predicted_Results` |
| 评分 | 负评（0）的 F1 |
| 代码 | [`code/01-sentiment`](../code/01-sentiment) · [`templates/01_文本二分类.py`](../templates/01_文本二分类.py) |

## 一、题目在问什么

一句话：**给一条顾客写的商品评论，预测这位顾客会不会推荐这件商品。**

目标列是 `Recommended IND`，取值只有 0 和 1：

- `1` = 推荐（我把这条当正评）
- `0` = 不推荐（负评）

题面给的理由很实在：评论数据量大，靠人工一条条看根本不现实，所以想用文本分类自动把差评捞出来。评分只看**负评那一类的 F1**，也就是 0 那一侧的 F1 —— 这个口径很关键，它意味着「全猜 1」的 F1 是 0，保底策略在这里会直接归零。

顺带说一个我当时没反应过来的事：这个练习的目录名叫「影评情感分析」，但数据里全是服装评论（`Clothing ID`、`Division Name`、`Class Name` 这些列）。**名字和数据不是一回事**，做题时只认字段，不认标题。我后面几篇练习也踩过类似的坑，标题只当索引看。

## 二、数据长什么样

两个 CSV，字段完全一致，只是测试集在本地这份里也带着答案列（正式测试集是没有的）。

`training.csv` 18789 行 × 10 列，`test-Ans.csv` 4697 行 × 10 列：

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `CID` | int64 | 18789 / 4697 | 评论序号，提交时的 ID 列 |
| `Clothing ID` | int64 | 1108 / 580 | 被评论的产品序号 |
| `Age` | int64 | 76 / 73 | 评论人年龄 |
| `Title` | object | 11465 / 3217 | 评论标题，**有缺失**（训练集里非空 15750 / 18789 行） |
| `Review Text` | object | 18106 / 4530 | 评论正文，**也有缺失**（训练集里非空 18110 / 18789 行） |
| `Positive Feedback Count` | int64 | 76 / 50 | 这条评论被多少人点过「有用」 |
| `Division Name` | object | 3 / 3 | 产品大类 |
| `Department Name` | object | 6 / 6 | 产品中类 |
| `Class Name` | object | 20 / 18 | 产品小类 |
| `Recommended IND` | int64 | 2 / 2 | **目标列**，1 = 推荐，0 = 不推荐 |

几件我扫数据时注意到的事：

1. **文本列大量缺失，而且缺得不均匀。** 直接 `dropna()` 会白扔掉上千行，所以绝对不能丢行，只能填空串。
2. `Class Name` / `Department Name` 这类**低基数类别列其实是强信号** —— 「这条评论是关于 Dresses 的」本身就比某些词更能说明倾向。我当时的做法是把它拼进文本里（粗糙但有效），模板里的做法更讲究：单独 one-hot 再 `hstack` 到特征矩阵上。
3. `Clothing ID` 有 1108 个不同取值，**高基数、几乎没信号**，别当数值特征喂进去。
4. 训练集 18789 行、测试集 4697 行，比例正好 4:1，加起来 23486 行，是原始数据集按比例切出来的。

## 三、思路

整个流程就是我那套标准动作，没什么花的：

```text
读数据 → 拼文本列 → 分词 → 去停用词 → TF-IDF → 朴素贝叶斯 → 写 CSV
```

具体决策和理由：

**1. 把三列拼成一个文本字段。**
`Title` + `Review Text` + `Class Name`，缺失填 `''`。标题和正文都是描述性文本，拼起来信息量最大；`Class Name` 拼进去算是个偷懒的特征工程 —— 反正 TF-IDF 会自己学权重，不需要我手动做 one-hot。

**2. 分词用 jieba，但只保留英文 token。**
数据是英文的，本来应该用 `nltk.word_tokenize`。我当时的写法是用 `jieba.lcut` 切，然后用正则把「纯英文、不含中文、不含数字」的 token 留下。逻辑上绕了一圈，效果上能用（jieba 对英文按空格和标点切，基本等价），但这是个当时没想明白的冗余设计，后面模板里直接换成了 `re.sub(r'\W', ' ', t)` 这种干净写法。

**3. 停用词表用现成的 `stop_words.csv`（2614 行）。**
注意它跟很多英文停用词表不一样，里面混了不少符号（第一行就是 `!`）。我在 `drop_stopwords` 里逐词比对过滤。

**4. 向量化和模型都选最笨的。**
`TfidfVectorizer` + `MultinomialNB(alpha=0.95)`。理由很直白：评论分类是典型的高维稀疏线性可分问题，朴素贝叶斯又快又稳，几秒钟出结果，先把提交文件拿到手再谈别的。

**5. 划分 20% 留出集看一眼分数，然后……**
这里是我当时最大的失误，见第五节。

## 四、关键代码

### 4.1 文本拼接：必须先 `fillna`

```python
train['text_new'] = (train['Title'].fillna('') + ' '
                     + train['Review Text'].fillna('') + ' '
                     + train['Class Name'].fillna(''))
```

不加 `fillna` 的话，pandas 遇到 NaN 会做字符串拼接后得到 `'nan'`，或者整行变 NaN，两种都很难查。**文本题只要有一列可能缺，就一定显式填空串。**

### 4.2 分词 + 去停用词：两步分开写

```python
def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(word)
        contents_clean.append(line_clean)
    return contents_clean, all_words
```

外层遍历文档、内层遍历词，同时把全量词收集到 `all_words` 里 —— 这个列表当时是想用来看词频分布的，虽然最后没用上。**去停用词一定要在分词之后、拼回字符串之前做**，否则停用词会被拼进长字符串里没法再切出来。

### 4.3 TF-IDF：两个参数值得单独说

```python
vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True,
                             smooth_idf=True, norm=None, max_df=0.7)
train_idf = vectorizer.fit_transform(df_train.train_words_list.values.tolist())
pre_idf = vectorizer.transform(df_test.test_words_list.values.tolist())
```

- 停用词在上面已经手工去过一遍了，这里又传了一次 —— 冗余，但无害。
- `max_df=0.7`：出现在 70% 以上文档里的词直接丢掉，相当于自动过滤「到处都是的口水词」。
- `norm=None` 关掉了 L2 归一化。对 `MultinomialNB` 来说这其实没什么必要，还改变了 TF-IDF 的尺度语义。现在回头看，这里应该保持默认的 `norm='l2'`。

关键点是**只用训练集 `fit`，测试集只 `transform`**，这一步我做对了。后来在别的练习里试过 train+test 一起 fit，字符级特征确实会更稳，但那是另一个话题（见模板文件顶部的注释）。

### 4.4 分层划分 + 模型

```python
x_train, x_test, y_train, y_test = train_test_split(
    train_idf, train_label.values.tolist(),
    test_size=0.2, random_state=1, stratify=train_label)

classifier = MultinomialNB(alpha=0.95)
classifier.fit(x_train, y_train)
```

`stratify` 不能省 —— 正负样本比例不均衡时（评论数据通常 80% 以上是正评），随机切可能切出偏差很大的验证集。

`alpha=0.95` 这个数是我随手试的值，**`GridSearchCV` 在文件头 `import` 了但全程没用**。现在知道应该老老实实搜一下 `alpha ∈ [0.1, 0.5, 1.0]`。这就是典型的「先跑通再优化」留下的尾巴。

### 4.5 输出：列名要跟题面一字不差

```python
ID_test = test['CID']
final_result = pd.DataFrame({'CID': ID_test, 'Predicted_Results': y_pred_test})
final_result.to_csv('Final_Results.csv', index=False)
```

`index=False` 别忘了，否则会多出一列无名索引。

## 五、踩的坑

### 坑 1：拿测试集的答案列当验证集，自己给自己打分

这是这篇里最该记的一条。文件最后两个 cell：

```python
classifier = MultinomialNB(alpha=0.95)
classifier.fit(train_idf, train_label)

y_pred = classifier.predict(pre_idf)
print(f"朴素贝叶斯-f1 : {round(f1_score(test_label, y_pred)*100,2)}%")
```

`test_label` 来自 `test['Recommended IND']` —— 那是**测试集的真实答案**。这个模型是在训练集全量上 fit 的，然后用测试集答案算分。这个数字除了能确认「代码没写错」，没有任何泛化意义，绝对不能当成成绩。

本地评估只有两种正确姿势：

- 从训练集切一块留出集，**模型不许见它**；
- 或者做 K 折交叉验证拿 out-of-fold 预测。

我当时只做到了前者（切了 20%），却在最后又用测试集答案打了一次分，属于自己骗自己。

### 坑 2：把 `Class Name` 拼进文本，稀释了真正的语义

`Class Name` 只有 20 个取值，拼进正文以后，一个单词 `Dresses` 会和几百个评论词一起做 TF-IDF。它带来的信号远不如单独 one-hot 干净。模板里后来改成了专门的 `CAT_COLS` 通道：

```python
# 只适合 Division/Department/Class 这类【取值少】的列；高基数会爆炸
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(pd.concat([cat_tr, cat_te]))
A = sparse.hstack([A, ohe.transform(cat_tr)]).tocsr()
```

**低基数类别列单独编码，别塞进文本里让字符特征去猜。**

### 坑 3：`pandas.util.testing` 这个 import

```python
import pandas.util.testing as tm
```

这行从头到尾没用到，而且在比较新的 pandas 里已经被移除了，直接 `ImportError`。**复制旧代码时一定要清一遍没用的 import**，否则换台机器第一行就崩。

### 坑 4：`norm=None` 和重复的停用词过滤

上面说过了：`TfidfVectorizer` 里传 `stop_words`，同时前面又手工去了一遍停用词，重复劳动。而且我把 `stopwords` 从 DataFrame 转成了 list —— 两处都传 list 能用，但如果只传一处，用 `stop_words='english'` 更省事。

## 六、小结

这次练习的价值不在于模型（朴素贝叶斯 + TF-IDF 是教科书第一页的内容），而在于**全流程跑通了**：

1. **文本列先 `fillna('')` 再拼**，这是文本题最容易翻车的地方；
2. **TF-IDF 只能 `fit` 训练集**，测试集只 `transform`；
3. **`stratify` 在不平衡二分类里是必须的**；
4. **提交文件的列名和 `index=False` 是拿分和不拿分的区别**；
5. **评估口径要跟评分口径对齐** —— 评分看负评 F1，本地评估就该只看负评那一类的 F1，而不是随手 `f1_score` 的默认值（默认 `average='binary'` 只对正类算，方向正好反了）。

后来我把这套流程整理成了 [`templates/01_文本二分类.py`](../templates/01_文本二分类.py)，它比这份 notebook 强在几个地方，都是实测踩出来的：

- **评分口径写进配置**：`SCORE_MODE` + `POS_LABEL` 显式声明「评分类是哪个」，选模型和调阈值都照着这个口径走，不会像这次一样评估方向搞反；
- **二分类阈值调优**：在 OOF 概率上扫 `0.05~0.95` 找 F1 最优点，而不是死守 `predict()` 的 0.5；
- **多数类基线**：先算一遍「全填多数类」的分数，打不过这个基线就说明特征做废了，得回头查；
- **候选池按 OOF 选**，而不是我这次直接钦定朴素贝叶斯。

一句话总结这次练习：**能跑出提交文件只是及格线，本地评估写对了才算真的会。**
