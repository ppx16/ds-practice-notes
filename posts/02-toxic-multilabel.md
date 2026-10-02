# 02 恶意评论识别与分级

> 一条评论可以同时是「恶毒」+「侮辱」+「仇视」，6 个标签各自独立 —— 多标签题的写法跟单标签完全不是一回事。

| | |
|---|---|
| 题型 | 文本 · 多标签（6 个标签各自独立 0/1） |
| 数据 | 训练 127657 行 × 8 列，测试 31914 行 × 8 列 |
| 提交 | `Final_Result.csv`，列：`id`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` |
| 评分 | 6 个标签各自的「1 类 F1」取平均（Macro-F1） |
| 代码 | [`code/02-toxic-multilabel`](../code/02-toxic-multilabel) · [`templates/02_多标签分类.py`](../templates/02_多标签分类.py) · [`templates/02_阈值调优实验.py`](../templates/02_阈值调优实验.py) |

## 一、题目在问什么

一句话：**判断一条社交平台评论属于哪几种不当言论，一个标签一个判断。**

跟上一篇最大的区别在于「答案不是一个数」：

| | 单标签二分类（上一篇） | 多标签（这一篇） |
|---|---|---|
| 一条样本的答案 | 1 个数（0 或 1） | 6 个数，**互相独立** |
| 可能的组合 | 2 种 | 2⁶ = 64 种 |
| 多数情况 | 非此即彼 | 大部分样本 6 个都是 0 |

6 个标签分别是：恶毒（`toxic`）、极度恶毒（`severe_toxic`）、猥亵（`obscene`）、威胁（`threat`）、侮辱（`insult`）、身份仇视（`identity_hate`）。题面特别强调：**一条评论可以一个标签都没有，也可以同时背上好几个标签**，所以这不是「6 选 1 的多分类」，而是「6 个独立的二分类」。

想清楚这一点，建模路线就定了：**拆成 6 个二分类问题，每个标签单独训一个模型**。这也是题面里评分口径暗示的方向 —— 它明确说「计算每个标签中 1 那一类的 F1，再取平均」，那就说明每个标签是各自评估的。

## 二、数据长什么样

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `id` | object | 127657 / 31914 | 16 位十六进制字符串，**是字符串不是数字** |
| `comment_text` | object | — | 评论文本（原始 CSV 里有换行符） |
| `toxic` | int64 | 2 | 目标 1：恶毒 |
| `severe_toxic` | int64 | 2 | 目标 2：极度恶毒 |
| `obscene` | int64 | 2 | 目标 3：猥亵 |
| `threat` | int64 | 2 | 目标 4：威胁 |
| `insult` | int64 | 2 | 目标 5：侮辱 |
| `identity_hate` | int64 | 2 | 目标 6：身份仇视 |

训练集 127657 行 × 8 列（`id` + 文本 + 6 个标签），测试集 31914 行同样 8 列，但正式测试集里没有那 6 个标签列。

另外有一份 `Stopwords.csv`，889 行 × 1 列，列名就是中文的 `停用词`，内容是英文停用词（`'d`、`'ll`、`'m`、`'re`、`'s` 这一堆缩写形式）。**列名是中文这一点坑过我**，读进来按 `stopwords['停用词']` 取，写成 `stopwords` 会 KeyError。

> 一个小提醒：我留档的本地提交文件是 20000 行，而训练/测试集本身是 127657 / 31914 行。**提交行数必须等于测试集行数**，所以那个 20000 行的存档只能当历史快照看，不能当成正确产出 —— 这也是我后来把「行数 = 测试集行数」写进交卷检查清单的原因。

### 数据本身的三个特点

我用 notebook 里那几个 cell 扫出来的：

1. **标签极度不平衡。** 把 6 列 `sum()` 一下就知道：`toxic` 是相对最多的，`obscene` / `insult` 其次，而 `threat` 和 `identity_hate` 的正例少得可怜 —— 相对 12 万行训练数据，正例占比是千分位级别。这是整个练习的核心难点。
2. **一条评论背几个标签的分布非常偏。** `rowsums = df_toxic.sum(axis=1)` 之后做 `value_counts()`，绝大多数样本是 0 个标签，1 个标签的次之，2 个以上的迅速衰减。所以「全填 0」在 accuracy 上会很好看，但在 Macro-F1 上会死得很惨 —— **这也是为什么评分选 Macro-F1 而不是 accuracy**。
3. **文本长度跨度极大。** `train.comment_text.str.len()` 的直方图（`bins=np.arange(0,5000,50)`）是典型的长尾：大量短评论，拖一根很长的尾巴。短文本会让 TF-IDF 更稀疏。

## 三、思路

### 3.1 为什么要拆成 6 个二分类

我一开始想过 `MultiLabelBinarizer` + `OneVsRestClassifier` 一把梭，但最后选了最直白的「for 循环 6 个标签」：

```python
for class_name in class_names:
    over_samples_x, over_samples_y = SMO.fit_resample(x_train, y_train[class_name])
    classifier = LogisticRegression(solver='sag', C=4)
    classifier.fit(over_samples_x, over_samples_y)
    submission[class_name] = classifier.predict(test_features)
```

好处是**每个标签可以有自己的处理策略**（独立的过采样、独立的阈值、独立的模型）。`OneVsRestClassifier` 也能做到，但不利于逐标签调试。对 6 个标签这种规模，手写循环完全够用，而且出问题时一眼能看出是哪个标签挂了。

### 3.2 用 SMOTE 过采样对付不平衡

`imblearn.over_sampling.SMOTE` 对每个标签的正例做合成过采样，把正负比例拉平再训。对 `threat` 这种正例只有几百条的标签，这是让模型「愿意预测 1」的关键一步。

⚠️ **这条经验后来被我自己推翻了**：现在的模板里 SMOTE 全删了，换成 `class_weight='balanced'`。原因很实际 —— 本机 `imblearn` 根本没装，那段代码一旦启用就 `ImportError` 直接崩，而且 SMOTE 在 12 万行 × 6 万维稀疏矩阵上做 `fit_resample`，内存和时间都不便宜。`class_weight='balanced'` 在逻辑回归上用几行代码就能达到类似效果。所以：**能装 imblearn 就用，装不上别硬上，`balanced` 足够对冲**。

### 3.3 特征：词级 + 字符级双通道

```python
word_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',
                                 analyzer='word', token_pattern=r'\w{1,}',
                                 ngram_range=(1, 2), min_df=3, max_features=30000)
char_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',
                                  analyzer='char', ngram_range=(1, 4),
                                  min_df=3, max_features=30000)
vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs=2)
```

字符级通道在这个任务上**不是锦上添花，是刚需**。因为不当评论里充斥着拼写变形：`sh!t`、`f u c k`、`y0u a holes` 这种。词级 tokenizer 会把它们切成完全不同的 token（或者整块丢掉），但 3~5 的字符 n-gram 能抓住 `s h !`、`f u c` 这种片段。词级通道抓正常表达，字符级通道抓变形 —— 两条腿走路。

`make_union` 把两个稀疏矩阵横向拼起来，等价于 `scipy.sparse.hstack`。

## 四、关键代码

### 4.1 清洗：把否定缩写展开，而不是删掉

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip(' ')
```

顺序很重要：**先把缩写展开成完整词，最后才 `\W` 把标点全干掉**。如果反过来先删标点，`can't` 就变成 `cant`，`n't` 这条规则永远匹配不到 —— 而「否定」在恶意评论判定里是强信号，不能丢。

`\W` 会把所有非单词字符换成空格，所以 `sh!t` 变成 `sh t`。这在字符级通道里反而是好事：`sh t` 的 3-gram 依然带信息。

### 4.2 用空格切词 + 去停用词，一步搞定

```python
stopwords = pd.read_csv('stopwords.csv')
stopwords = stopwords['停用词'].tolist()

def f(x):
    current_segment = x.split(' ')          # 分词
    current_segment = [s for s in current_segment if s not in stopwords]
    return " ".join(current_segment)

train['comment_text'] = train['comment_text'].apply(f)
test['comment_text']  = test['comment_text'].apply(f)
```

因为上一步的 `\W` 已经把标点变成了空格，这里直接 `split(' ')` 就是分词，不需要 nltk。`df.apply()` 比 for 循环快得多，12 万行也就几秒钟。

注意 `stopwords['停用词']` 这个中文列名 —— 就是这里。

### 4.3 特征在 train+test 上一起 fit

```python
train_text = train['comment_text']
test_text  = test['comment_text']
all_text   = pd.concat([train_text, test_text])

vectorizer.fit(all_text)
train_features = vectorizer.transform(train_text)
test_features  = vectorizer.transform(test_text)
```

这是跟上一篇（01）的**刻意不同**。上一篇我严格只 `fit` 训练集。这里我改成了 train+test 一起 `fit` 词表，理由是：字符级 n-gram 的碎片（比如 `sh t`）很可能只在测试集里出现，只按训练集建词表会让这部分信息全丢。

代价是理论上有一点点信息泄漏（测试集的词频影响了 IDF）。这个练习里我认了，因为**候选池是按本地留出集选的，不是按测试集选的**，所以不会出现「靠测试集词表把分数刷上去」的自欺。

后来证明这个选择是对的 —— 模板文件顶部的注释里记着：字符特征如果只按训练集 fit，某个练习的 accuracy 会从 0.936 掉到 0.885。**tf-idf 的拟合范围是个真实存在的开关，不是玄学。**

### 4.4 逐标签过采样 + 训练 + 计数

```python
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
SMO = SMOTE(random_state=3)

submission = pd.DataFrame.from_dict({'id': test['id']})

for class_name in class_names:
    over_samples_x, over_samples_y = SMO.fit_resample(x_train, y_train[class_name])
    classifier = LogisticRegression(solver='sag', C=4)
    classifier.fit(over_samples_x, over_samples_y)
    submission[class_name] = classifier.predict(test_features)
```

三个细节：

- **`y_train[class_name]` 是 Series，不是 DataFrame** —— 传二维标签给 SMOTE 会报错，多标签题在这里最容易翻车；
- **`solver='sag'`** 是随机平均梯度下降，在高维稀疏数据上比默认的 `lbfgs` 快很多；
- **`C=4`** 比默认的 1.0 更宽松，让模型在过采样后的平衡数据上更敢下判断。

### 4.5 输出 7 列 0/1

```python
submission.to_csv('Final_Result.csv', index=False)
```

`submission` 是从 `{'id': test['id']}` 建起来的 DataFrame，循环里逐列挂上 6 个标签，最后一次性落盘。**这种「先建骨架、逐列填」的写法比在循环里反复 `pd.concat` 清楚得多。**

## 五、踩的坑

### 坑 1：提交文件名是单数 `Final_Result.csv`

代码里写的是 `Final_Result.csv`（没有 s），而上一篇（01）和下一篇（05）写的是 `Final_Results.csv`（有 s）。**这种一个字母的差别，题面要求是什么就必须是什么。** 我后来在交卷清单里把「文件名和列名逐字比对」列成了固定动作，就是因为这类事故。

### 坑 2：本地测试集读不进来 —— 编码问题

原始 CSV 是 `ISO-8859-1`，不是 UTF-8。这个必须显式指定：

```python
train = pd.read_csv('../data/training.csv', encoding="ISO-8859-1")
test  = pd.read_csv('../data/test-Ans.csv',  encoding="ISO-8859-1")
```

我是先看到 `UnicodeDecodeError` 才反应过来加 `encoding` 的。现在模板里的做法是写个 `read_table()` 依次试 `utf-8-sig → gb18030 → latin-1`：

```python
for enc in ['utf-8-sig', 'gb18030', 'latin-1']:
    try:
        return pd.read_csv(path, encoding=enc, **kw)
    except (UnicodeDecodeError, UnicodeError):
        continue
return pd.read_csv(path, encoding='latin-1')
```

**`latin-1` 是永不报错的兜底编码**（任何字节序列都是合法的 latin-1），所以放在最后一定读得进来，最多是乱码。这个函数后来在我每个文本题里都复用了。

### 坑 3：本地 `test-Ans.csv` 里带着答案，结果有两种不同的算法

notebook 最后两个 cell 分别这样算：

```python
# 方式 A：在训练集上切 20% 留出，用留出集算 F1
f1 = f1_score(y_test[class_name], classifier.predict(x_test))
print('Macro-F1 is {}'.format(np.mean(scores_test)))

# 方式 B：全量训练，用测试集答案算 F1
f1 = f1_score(test[class_name], classifier.predict(test_features))
print('Macro-F1 is {}'.format(np.mean(scores)))
```

方式 A 是**有意义的**本地评估（模型没见过 `x_test`）。方式 B 是把测试集答案拿来打分，**没有泛化意义**，只能用来确认代码跑通。这跟上一篇（01）踩的是同一个坑。

多标签题里的额外教训是：**不要把两种口径的数字混在一起看**。同一份代码打印两个 Macro-F1，很容易记混哪个是哪个。现在我会在变量名上区分：`val_f1`（留出集）和 `test_f1`（本地答案，只用来 sanity check）。

### 坑 4：6 个标签里 `threat` 最难受

`threat` 和 `identity_hate` 的正例在 12 万行里少到千分位级别。过采样之后模型能学到东西了，但**每次运行结果波动都不小**（`SMOTE(random_state=3)` 是固定的，所以至少可复现）。

真正救这两个标签的是**逐标签调阈值**，这条路我当时没走通，后来在 [`templates/02_阈值调优实验.py`](../templates/02_阈值调优实验.py) 里补上了：留 20% 验证集，对每个标签单独扫 `0.05~0.95` 找个 F1 最高的阈值，再拿全量重训、用这个阈值去卡测试集概率：

```python
def run(features, name, tune=True):
    Xtr, Xte = features
    for col in LABELS:
        ...
        pv = clf.predict_proba(X_b)[:, 1]      # 验证集概率
        thr = 0.5
        if tune:
            best_t, best_f = 0.5, -1.0
            for t in np.arange(0.05, 0.96, 0.01):
                f = f1_score(y_b, (pv >= t).astype(int), zero_division=0)
                if f > best_f: best_f, best_t = f, t
            thr = best_t
        clf_full.fit(Xtr, y)
        sub[col] = (clf_full.predict_proba(Xte)[:, 1] >= thr).astype(int)
```

这份实验脚本自己的结论（写在文件开头）是：**硬 0.5 的基线 Macro-F1 ≈ 0.582，调阈值之后，稀有标签（`severe_toxic` / `threat` / `identity_hate`）涨得最明显**。同一个文件里后来记的改进版结论是 0.582 → 0.635，涨的那部分主要来自两件事：逐标签阈值 + 补上 `char_wb(3,5)` 字符通道。

> 这两个数字是文件里自己记下来的实验结论，不是这次 notebook 跑出来的。notebook 这份我只跑出了提交文件，本地分数当时没有记进文件。

**为什么高阈值对稀有标签有用**：正例极少时，模型给的概率普遍偏低，`>= 0.5` 会把大量真阳漏掉；把阈值降到 0.3 左右，recall 上去，F1 往往跟着上去。反过来，`toxic` 这种正例多的标签，最优阈值可能接近甚至高于 0.5。

### 坑 5：`Stopwords.csv` 里都是缩写残片

889 行停用词里开头就是 `'d`、`'ll`、`'m`、`'re`、`'s`。但我在清洗那一步已经把 `'s` 展开了（`\'s` → `is`），所以这份停用词表里的一大半根本匹配不上了 —— **两份处理在打架**。

正确的顺序应该是二选一：

- 要么在清洗阶段展开缩写，然后停用词表用正常的英文停用词；
- 要么清洗阶段不动缩写，让停用词表去过滤这些残片。

两个都做，等于停用词表白读了一半。

## 六、小结

这是四个练习里我收获最大的一个，因为它逼我把「多标签 ≠ 多分类」这件事真正想清楚了：

1. **多标签拆成 N 个独立二分类**，每个标签自己的正例比例、自己的最优阈值、甚至自己的模型；
2. **降采样/过采样要逐标签做**，`y` 必须是 Series 不能是 DataFrame；
3. **字符级 n-gram 对「拼写变形」类文本是刚需**，词级通道抓不到 `sh!t` 这种；
4. **清洗顺序：先展开缩写，再删标点，最后去停用词** —— 顺序错了规则就失效；
5. **Macro-F1 是多标签题的标准评分**，因为它对稀有标签一视同仁，`accuracy` 会被「全填 0」刷爆；
6. **本地评估只能信留出集 / OOF**，测试集答案打出来的分只能当 sanity check。

最重要的方法论转变：**这次我用一个固定的 `for` 循环硬编码了 6 个标签名和流程，下次换个标签数就得改代码。** 所以后来写了 [`templates/02_多标签分类_通用版.py`](../templates/02_多标签分类_通用版.py)，把标签形态（多列 one-hot vs 单列逗号分隔）、中英文分流、逐标签阈值全部做成配置项，只改 `CONFIG` 那几行就能换题。那份模板里还有一条我这次没做、但确实该做的事：

```python
def guess_label_cols(df, id_col, text_col):
    cols = []
    for c in df.columns:
        if c in (id_col, text_col):
            continue
        vals = set(df[c].dropna().unique())
        if len(vals) == 2 and vals <= {0, 1, '0', '1'}:
            cols.append(c)
    return cols
```

**自动认出所有 0/1 列当标签**，省得每次手抄标签名抄错。手抄 6 个英文标签名，抄错一个就白跑一轮。

一句话总结：**多标签题赢在「逐标签」这三个字上 —— 逐标签过采样、逐标签调阈值、逐标签看 F1。**
