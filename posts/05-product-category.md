# 05 中英商品标题分类

> 24 个中文类别、没有空格、描述里还混着英文型号 —— 中文多分类的关键不是换模型，是把 TF-IDF 从「词级」换成「字符级」。

| | |
|---|---|
| 题型 | 文本 · 多分类（单标签，24 类） |
| 数据 | 训练 3342 行 × 3 列，测试 1114 行 × 3 列 |
| 提交 | `Final_Results.csv`，列：`Product_ID`, `Predicted_Product_Category` |
| 评分 | Micro-F1（单标签多分类下等价于 Accuracy） |
| 代码 | [`code/05-product-category`](../code/05-product-category) · [`templates/05_中英多分类.py`](../templates/05_中英多分类.py) |

## 一、题目在问什么

一句话：**给一段商品描述，预测它属于 24 个商品类别中的哪一类。**

跟前面两个练习的区别一眼能看出来：

| | 01 / 02 | 05 |
|---|---|---|
| 语言 | 英文 | **中文为主，混英文和数字** |
| 类别数 | 2 类 / 6 个独立标签 | **24 类，互斥** |
| 提交的值 | 0/1 数字 | **中文类别名原文** |

最后一行是这道题最容易丢分的地方。题面要求提交的 `Predicted_Product_Category` 必须是**文字**，不能是数字编码。我一开始把类别映射成了 0~23 训练，最后差一点就忘了 `inverse_transform` 回去 —— 交一堆数字上去直接 0 分。

评分用 Micro-F1。单标签多分类里，Micro-F1 的定义是「所有类的 TP 总和 / 所有类的 (TP+FP) 总和」，因为每个样本只属于一个类，这个式子化简下来就是 `正确数 / 总数` —— **数值上等于 Accuracy**。所以本地评估用 `accuracy_score` 和 `f1_score(average='micro')` 是同一个数。

## 二、数据长什么样

两个 Excel 文件，不是 CSV。

`training.xlsx` 3342 行 × 3 列，`test-Ans.xlsx` 1114 行 × 3 列：

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `Product_ID` | int64 | 3342 / 1114 | 产品序号，提交时的 ID 列 |
| `Product_Category` | object | **24 / 24** | **目标列**，中文类别名 |
| `Product_Description` | object | 3319 / 1110 | 商品描述正文，**有缺失**（训练集 3339 行非空，测试集 1113 行非空） |

关于目标列，有几个数字值得记一下：

- 训练集 3342 行，24 个类别 → **平均每类只有 139 条**，而且分布不均匀，属于典型的小样本多分类；
- 测试集本地这份 `test-Ans.xlsx` 里也带着 `Product_Category` 列（1114 行全部非空）。这正是它文件名叫 `-Ans` 的原因，也是第五节那个坑的来源；
- 我最终交出的 `Final_Results.csv` 里只出现了 22 个不同类别，另一个历史提交文件里有 23 个。**模型没把所有类别都预测出来是很正常的事** —— 小样本多分类里，稀有类别经常一个都没被预测到。

描述文本的几个特点（读了前几行就知道）：

1. **中文，几乎没有空格。** 这是最关键的一点，直接决定了特征怎么做（见第三节）。
2. **中英混排、带数字。** 描述里会出现 `LINQtoU`、`iPhone`、`128GB`、`5-99岁`、`✅3秒即热` 这种东西。
3. **带 emoji 和特殊符号。** 上面那个 `✅` 就是真实的例子，还有换行符 `\n`、括号、全角逗号。
4. **长度差异极大**，从十几个字到几百个字都有。

另外有一份 `stopwords.txt`，2614 行，是英文停用词表（内容跟前两个练习的停用词表是同源的，第一行是 `!` 这种符号）。

## 三、思路

### 3.1 中文题的第一个决策：词级还是字符级

**中文没有空格，所以 sklearn 默认的 `token_pattern=r'\w{1,}'` 会把一整句话切成一个 token** —— 词级 TF-IDF 在中文上基本等于没做特征。这是中文文本题最容易踩的暗坑，而且不报错，只是分数很难看。

有两条路：

**路 A：先分词（jieba），再按词做 TF-IDF。**
我这份 notebook 走的就是这条路。`jieba.posseg.cut()` 分词 + 词性标注，按词性过滤掉数词（`m`）、字符串（`x`）、助词（`ul`），再去停用词。

**路 B：不分词，直接做字符 n-gram（`analyzer='char_wb'`）。**
这是我现在推荐的做法。

我当时的想法很朴素：中文分类当然要分词。**后来实测才发现路 B 更好，而且省事得多。** 现在 [`templates/05_中英多分类.py`](../templates/05_中英多分类.py) 里走的是路 B：

```python
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4),
                        sublinear_tf=True, min_df=2, decode_error='ignore')
```

三个原因：

1. **`char_wb` 天然处理中文。** 中文的词大多 1~2 个字，1~4 的字符 n-gram 能覆盖绝大部分词，还顺便抓到了词边界。
2. **`ngram_range` 必须含 1。** 这条是实测出来的：`(2,4)` 比 `(1,4)` 差一大截。因为中文里「枕头」「耳机」「吉他」这种单字/双字词很多，砍掉单字就砍掉了主要信号。
3. **不用装 jieba，不用调词性过滤规则**，少一层依赖少一层坑。

模板文件顶部记着这条路线的实测对比（同一个测试集）：

| 配置 | Micro-F1 |
|---|---|
| 词级 + `char_wb(2,4)` 双通道 + XGBoost | 0.4532 |
| jieba 分词 + 去停用词 + 词级 TF-IDF | ~0.45 |
| **`char_wb(1,4)` 单通道 + LinearSVC** | **0.6427** |

那三行是模板文件里自己记下来的实测结论。差了 0.19，**不是调参调出来的，是特征换对了**。

### 3.2 第二个决策：清洗时保留英文和数字

中文题的清洗很容易走过火。我试过「只留中文字符」，结果描述里的 `iPhone`、`128GB`、`LINQtoU` 全被抹掉了 —— 而**这些东西恰恰是最强的类别信号**（「128GB」几乎直接指向科技用品）。

正确的清洗是**只留中文 + 英文 + 数字，去掉标点符号和特殊字符**：

```python
def clean_text(t):
    if not isinstance(t, str):
        return ''
    return re.sub(r'[^一-鿿a-zA-Z0-9]', ' ', t).strip()
```

这个正则的意思是「把**不在**中文区间、不在 a-z/A-Z、不在 0-9 里的字符，全部换成空格」。对比一下实测差距（同样来自模板文件里的记录）：只留中文是 0.630，中英数字都留是 0.643。差得不多，但方向很明确 —— **能留下的信号就别扔**。

### 3.3 第三个决策：模型用不用树

我这份 notebook 里试了三个模型：朴素贝叶斯（GridSearch 选 `alpha`）、随机森林（1000 棵树，配 SMOTE 和 `class_weight='balanced'` 两个版本）、XGBoost（360 棵、深度 3）。

最后交出去的是**朴素贝叶斯**。

现在回头看，这个选择没问题，但结论更极端了：**高维稀疏的 TF-IDF 上，线性模型（LinearSVC）通常又快又准，树模型反而吃亏。** 模板里记的对比是 XGBoost 0.577 vs LinearSVC 0.643。原因不难理解：字符 n-gram 特征维度轻松上十万，树模型在这么稀疏的空间里分裂效率极低，而线性模型（SVC / 逻辑回归 / 朴素贝叶斯）天生就是干这个的。

所以现在的模板里，中文多分类的第一选择是：

```python
model = LinearSVC(C=1.0)
model.fit(A, y)
```

### 3.4 第四个决策：类别编码成数字再映射回来

sklearn 的分类器要求 `y` 是数字或有序标签，中文类别名不能直接喂（能喂但要额外编码）。所以：

```python
list_columns = df_train['Product_Category'].unique()
list_nums = np.arange(len(list_columns))
cate_dict = dict(zip(list_columns, list_nums))   # 中文 -> 数字
num_dict  = dict(zip(list_nums, list_columns))   # 数字 -> 中文

df_train['target'] = df_train['Product_Category'].apply(lambda x: cate_dict.get(x))
```

注意最后用了 `.get()` 不是 `[x]`。**这是一个真实的隐患**：如果测试集里出现训练集没有的类别，`cate_dict[x]` 会 KeyError 崩掉，而 `.get(x)` 会安静地返回 `None`，把问题推到后面。两种都不好，但 `.get()` 至少不会中断流程。

更规范的做法是用 `LabelEncoder` + `inverse_transform`（模板里就是这么写的），它会自己维护 `classes_` 数组，不用手工维护两个字典：

```python
le = LabelEncoder()
y = le.fit_transform(train[TARGET].astype(str))
...
pred_label = le.inverse_transform(pred)
```

## 四、关键代码

### 4.1 分词 + 词性过滤（这份 notebook 的做法）

```python
import jieba.posseg as pseg

def des_clean(s):
    segs = pseg.cut(s)
    final = ''
    for seg, flag in segs:
        if seg not in stopwords:                       # 去停用词
            if flag != 'm' and flag != 'x' and flag != 'ul':   # 去数词/字符串/助词
                final += ' ' + seg
    return final
```

`pseg.cut()` 返回 `(词, 词性)` 对。这里的过滤规则是：去掉数词 `m`、非语素字 `x`、助词 `ul`。逻辑上说得通，但**词性标签是 jieba 猜的，不一定靠谱**，而且我为了「过滤数词」把 `128GB` 里的数字也一起干掉了 —— 跟 3.2 说的「数字是信号」正好矛盾。

**这就是路 A 的隐性成本：每加一条过滤规则，都可能顺手扔掉一个有用信号。**

### 4.2 加载停用词：`remove('')` 这个小动作不能漏

```python
stopwords = set()

with open('stopwords.txt', 'r', encoding='utf-8') as sw:
    [stopwords.add(line.strip()) for line in sw.readlines()]

stopwords.remove('')
len(stopwords)
```

读文件的时候，空行 `strip()` 之后变成空字符串 `''`，它会被加进集合里。然后分词时如果有任何空片段，就会因为「在停用词表里」被过滤掉 —— 逻辑上没坏处，但 `''` 出现在停用词表里是脏数据，所以显式 `remove('')` 掉。**顺便用 `len(stopwords)` 确认一下到底读进来多少个词**，比 0 大很多才说明读对了（我这里是 2614 行去掉空行）。

### 4.3 本地留出集：先切分，再 fit 向量器

```python
X_train, X_test, y_train, y_test = train_test_split(
    df_train.Product_Description_clean, df_train.target,
    test_size=0.25, random_state=33)

tv = TfidfVectorizer(sublinear_tf=True, decode_error='ignore')
x_train = tv.fit_transform(X_train)
x_test  = tv.transform(X_test)
```

顺序对了：**先切数据，再在训练那部分上 `fit` 向量器**。如果先在全量上 `fit` 再切分，IDF 就偷看了验证集。

`decode_error='ignore'` 是给中文编码问题兜底的 —— 万一某条描述编码坏了，直接跳过而不是整个崩掉。

### 4.4 用 `neg_log_loss` 搜朴素贝叶斯的 `alpha`

```python
mnb = MultinomialNB()
MNB_params = {'alpha': [0.02, 0.05, 0.1, 0.5]}
MNB_grid = GridSearchCV(mnb, MNB_params, cv=5, verbose=1,
                        scoring='neg_log_loss', n_jobs=-1)
MNB_grid.fit(x_train, y_train)
```

**这里是这次练习里一个真实的错配**：评分口径是 Micro-F1，但我用 `neg_log_loss` 去搜超参。两者相关但不一致 —— 对数损失关心概率校准，F1 关心阈值处的硬分类结果。搜索目标跟最终指标不一致，选出来的 `alpha` 不一定是 F1 最优的。

正确做法是 `scoring='f1_micro'`（或者自定义 scorer）。**「调参优化的指标必须等于评分的指标」**，这一条我在后面几个练习里才真正养成习惯。

`n_jobs=-1` 开满所有核心，`cv=5` 五折，`alpha` 四个候选值 —— 总共 20 次拟合，对这个数据量（3342 行）几秒钟就完事。

### 4.5 修掉那个最贵的 bug：把预测结果映射回中文

```python
df_test['label'] = y_pred
df_test['Predicted_Product_Category'] = df_test['label'].apply(lambda x: num_dict.get(x))
df_test[['Product_ID', 'Predicted_Product_Category']].to_csv(
    'Final_Results.csv', encoding='gb18030', index=False)
```

两件事：

1. **`num_dict` 把数字映射回中文类别名**，这一步绝不能忘；
2. **`encoding='gb18030'`**，中文 CSV 用这个编码。这个不是随便选的 —— 参考数据本身就是 gb18030，用 `utf-8` 写出去在 Excel 里打开会乱码，用 `utf-8-sig` 又跟参考格式对不上。**跟参考文件保持同一个编码**是最稳的做法。

最后只取 `[['Product_ID', 'Predicted_Product_Category']]` 两列落盘，中间那些辅助列（`target`、`label`、`Product_Description_clean`）不写进提交文件。

## 五、踩的坑

### 坑 1（最贵的一个）：拿测试集的答案列当本地评估集

这份 notebook 最后几步是这样的：

```python
# 产生最终输出结果
X_train = df_train.Product_Description_clean
y_train = df_train.target
X_test  = df_test.Product_Description_clean
y_test  = df_test.target          # ← 这就是测试集的真实答案
...
y_pred = MNB_grid.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Micro F1:', f1_score(y_test, y_pred, average='micro'))
```

`df_test.target` 是在前面第 8 个 cell 里算出来的：

```python
df_test['target'] = df_test['Product_Category'].apply(lambda x: cate_dict.get(x))
```

也就是说，**本地这份 `test-Ans.xlsx` 里带着答案列，我把它读进来算了 F1。** 这个数字只能说明代码没写错，**不能当成泛化成绩**，更不能拿去跟别人的分数比。

这跟前两篇（01、02）是同一个坑，但这次更隐蔽：因为在写「最终输出」那一段里顺手算了一次分，看起来像是流程的一部分，实际上是把答案泄露给了评估。

**处理办法**：本地评估永远只认两种数 —— 训练集内部的留出集/OOF，或者你明确知道「这是本地答案文件，只看代码是否跑通」。我在文件里现在会写成 `test_f1_sanity` 这种变量名，强迫自己记住它是什么。

### 坑 2：`test-Ans.xlsx` 和考场用的 `test.xlsx` 不是一回事

本地这份测试集带 `Product_Category` 列，但正式测试集是没有的。这带来一个隐蔽的坑：

```python
list_columns = df_train['Product_Category'].unique()   # 训练集的 24 类
cate_dict = dict(zip(list_columns, list_nums))
df_test['target'] = df_test['Product_Category'].apply(lambda x: cate_dict.get(x))
```

如果测试集里出现了训练集没见过的类别，`.get()` 会返回 `None`，然后在后面 `num_dict.get(None)` 也是 `None`，最后在评估时被当成一个「空类别」混进 confusion matrix。**严格说这是数据泄漏**：你用了测试集的答案列去构造评估目标。

代码里应该加一句断言：

```python
unknown = set(df_test['Product_Category']) - set(df_train['Product_Category'])
assert not unknown, f'测试集出现未见过类别: {unknown}'
```

**「本地有答案」这件事本身不是问题，问题是你别把它接进流程里。** 正确做法是读进来先 `drop(columns=['Product_Category'])`，模拟考场环境。

### 坑 3：SMOTE 把 3342 行撑成 24 类的平衡集

```python
sm = SMOTE()
x_new, y_new = sm.fit_resample(x_train, y_train)
rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1).fit(x_new, y_new)
```

24 类里最少的类可能只有几十条，SMOTE 会把它合成到跟最多类一样多。3342 行的训练集（切完 25% 后剩 2500 行）过采样之后会膨胀好几倍，然后还要训 1000 棵随机森林 —— **在考试的时间限制下这是纯浪费**。

而且对文本的 TF-IDF 稀疏特征做 SMOTE 插值，物理意义很可疑：两个稀疏向量插值出来的「新样本」可能同时包含两类的关键词。所以现在我对文本题的取舍很明确：

- **不平衡不严重** → 什么都不做，或者用 `class_weight='balanced'`；
- **不平衡严重（如上一篇 02 的 `threat` 标签）** → 过采样可以做，但优先考虑**调阈值**，成本更低、效果更直接。

### 坑 4：`stopwords.txt` 和 `stopwords_.txt` 两个文件

`code/05-product-category/code/` 底下有两份停用词：`stopwords.txt`（2614 行）和 `stopwords_.txt`（小得多的一份）。代码里读的是前者。**仓库里留着两个版本是历史遗留**，但不写清楚的话，下次打开会不知道该用哪个。我在这篇里明确记一下：**这份 notebook 用的是 `stopwords.txt`**。

### 坑 5：`GridSearchCV` 的 `scoring` 跟最终指标不一致

前面 4.4 说过了，这里再强调一次，因为它**不报错、不警告，只是悄悄让你选错参数**：

```python
MNB_grid = GridSearchCV(mnb, MNB_params, cv=5, scoring='neg_log_loss', ...)
...
print('Micro F1:', f1_score(y_test, y_pred, average='micro'))
```

搜的是对数损失，量的是 Micro-F1。改成 `scoring='f1_micro'` 就一致了。

## 六、小结

这道题教我的事，一半在中文文本上，一半在工作方法上。

**中文文本部分：**

1. **中文没有空格，词级 TF-IDF 直接废掉** —— 这是中文题的头号暗坑，不报错但分数极低；
2. **`analyzer='char_wb'` + `ngram_range=(1,4)` 是中文分类的性价比之王**，比 jieba 分词 + 词级 TF-IDF 明显好，还少一层依赖；
3. **`ngram_range` 必须含 1**，(2,4) 会砍掉单字词这个主要信号；
4. **清洗只留中文+英文+数字**，`iPhone`、`128GB` 这类混排内容是强类别信号，别一起洗掉；
5. **高维稀疏特征上线性模型（LinearSVC）优于树模型**，别一看多分类就上 XGBoost。

**工作方法部分：**

6. **提交的是中文类别名，不是数字编码** —— 用 `LabelEncoder.inverse_transform` 而不是手写两个字典，少一个漏映射的机会；
7. **中文 CSV 的编码跟参考文件对齐（gb18030）**；
8. **调参的 `scoring` 必须等于评分指标**，否则搜出来的超参不是你要的；
9. **本地带答案的测试集只能用来 sanity check**，一定要先 `drop` 掉答案列再进流程。

现在这道题的标准解法就是 [`templates/05_中英多分类.py`](../templates/05_中英多分类.py)，全篇 85 行，核心只有三句：清洗只留中英数字 → `char_wb(1,4)` 的 TF-IDF → `LinearSVC`。**从 100 多行的 notebook 压到 85 行，分数反而更高** —— 这是我做这套模板最大的收获：删掉的每一行都是在清洗阶段引入的噪声。

一句话总结：**中文文本分类，先把「怎么切词」这件事解决掉，再谈模型。**
