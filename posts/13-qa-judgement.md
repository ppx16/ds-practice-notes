# 13 问答判定

> 这题我押的是「不训分类头，直接让生成模型吐 True/False」——省事，但**零样本没有打赢多数类基线**，这是我做完才知道的。

| | |
|---|---|
| 题型 | 生成式 / 判定（是 / 否） |
| 数据 | 训练与测试 CSV 读取失败（编码问题），只能确认提交文件 7,541 行 × 2 列 |
| 提交 | `考生姓名_results.csv`，列：`question_id`, `Predicted_Results` |
| 评分 | — |
| 代码 | [`code/13-qa-judgement`](../code/13-qa-judgement) · [`templates/13_flan_t5生成式.py`](../templates/13_flan_t5生成式.py) |

> 关于「评分」这一格：我在代码和模板里**都没有找到这题的评分说明**，所以就留了 `—` 而不是猜一个
> 指标填上去。从提交列名（`Predicted_Results`）和标签是布尔值来看，多半是按准确率算的，
> 但「多半」不能当依据写进笔记。

## 一、题目在问什么

给一段材料（`passage`）和一个问题（`question`），判断**这段材料能不能回答这个问题**。
输出是布尔判定，交两列：问题编号 + 判定结果。

这跟第 9 题的「阅读理解多选」长得像，但性质完全不同：

- 第 9 题是**四选一**，四个候选项本身就在提示你该往哪找；
- 这题是**是非判定**，模型要自己读完整段材料，然后回答「这段材料够不够回答问题」。

`code/13-qa-judgement/` 下面**只有一个文件**（`qa-judgement.ipynb`），
没有第二个版本、没有脚本、没有对比实验。所以这篇复盘能讲的东西比另外几篇少，
我也不打算把它撑长——**说清楚的短，好过编出来的长。**

## 二、数据长什么样

先说结论：**这题的数据字段我没能完整拿到。**

字段卡片里两个数据文件的记录是这样的：

| 文件 | 结果 |
|---|---|
| 训练 CSV | 读取失败（编码错误） |
| 测试 CSV | 读取失败（编码错误） |

失败原因是自动抽取脚本用 `gb18030` 去解码，在中途撞到一个非法字节。
所以**这两张表真实的列名、类型、行数，我在这里给不出来**——
下面写的字段是从代码里「反推」出来的，跟真实文件的字段名可能有出入。

### 提交文件（7,541 行 × 2 列）

这一张是能确认的：

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `question_id` | int64 | 7,541 | 问题编号，唯一 |
| `Predicted_Results` | bool | 2 | 判定结果 |

关于最后那一列有个坑要提醒自己：卡片上 `dtype` 显示成 `bool`、`样例` 显示成 `True`，
**这只是 pandas 把读进来的文本解析成了布尔值，并不代表文件里写的就是 `True` 而不是 `TRUE`。**
大小写这一项在卡片上是看不出来的，只能直接打开文件看原字符。

### 代码里用到的字段（反推，未验证）

| 出现在哪 | 字段 | 依据 |
|---|---|---|
| 训练集 | `passage`, `question`, `answer` | `format_input` 取 `passage`/`question`；自检时拿 `answer` 当真值 |
| 训练集 | `question_id`, `title`, 以及一个 `Unnamed: 0` 索引列 | notebook 里 `train_i` 的表格打印 |
| 测试集 | `question_id`, `passage`, `question` | `format_input` + 最后拼提交文件 |

两处必须记下来的不一致：

1. **标签列名对不上。** notebook 里用的是 `answer`，
   而模板 [`templates/13_flan_t5生成式.py`](../templates/13_flan_t5生成式.py) 里读的是
   `train['Predicted_Results']`。这两个名字不可能都对。
   我判断是当时写模板时凭记忆填的，**照抄之前一定要先把真实的列名打印出来**。
2. **测试集行数取不出来。** 提交文件是 7,541 行，但 notebook 里存下来的预测分布
   （`yes 717 / True 516 / no 336 / Fal 228 / false 78 / true 11`，合计 1,886）
   只覆盖了一部分——那次运行看起来是被中断过。所以「测试集到底多少行、
   完整跑一遍要多久」这两个问题，我从材料里答不出来。

## 三、思路

思路只有一句话：**不训分类头，把判定当成一个生成任务。**

常规做法是取一个预训练模型（BERT 之类），在 `[CLS]` 上接一个二分类头，
拿训练数据微调。这题我走的是另一条路：

```text
prompt = 固定引导语 + 材料 + 问题
      ↓
flan-t5-base（encoder-decoder 生成模型，零样本，不微调）
      ↓
生成一个 token："True" 或 "False"
      ↓
字符串归一化 → 提交
```

选这条路的理由：

- **零样本**。不训练就不用划分验证集、不用调学习率、不用管显存够不够跑 batch。
  训练数据只用来做一次自检。
- **代码极短**。整条流水线就是「拼 prompt → generate → 解析字符串」三步，
  比搭分类头、写 Dataset、写 Trainer 短一个数量级。
- **`flan-t5-base` 是 instruction-tuned 的**，天生就会按指令输出短答案，
  不用额外教它「请回答 True 或 False」。

代价也很明显，而且我后来才真正意识到：**零样本的上限就摆在那儿，你没有用训练数据去提升它。**
训练数据在这条路线里唯一的作用是「事后告诉你它有多差」。

## 四、关键代码

### 4.1 拼 prompt

```python
from string import Template

preamble = ('Use the articles about the question, '
            'Answer the following question by outputting the True or False')

template = Template('$preamble\n\n articles : $article\n\n question : $question')

def format_input(df, idx):
    return template.substitute(preamble=preamble,
                               article=df.loc[idx, 'passage'],
                               question=df.loc[idx, 'question'])
```

用 `string.Template` 而不是 f-string，是为了让 prompt 模板能像配置一样独立拿出来改。
prompt 的结构是 **引导语 → 材料 → 问题**，顺序很关键，下一节会看到它带来的麻烦。

### 4.2 生成 + 字符串归一化

```python
inputs  = tokenizer(format_input(train, i), return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=1)      # 只要 1 个 token
answer  = tokenizer.batch_decode(outputs, skip_special_tokens=True)

train_i['Predicted_Results'] = train_i['answers'].replace(
    {'yes': 'TRUE', 'True': 'TRUE', 'true': 'TRUE',
     'no':  'FALSE', 'Fal': 'FALSE', 'false': 'FALSE'})
```

这个归一化字典是整个 notebook 里信息量最大的一行：

- 字典里同时有 `yes/no`、`True/true`、`false` —— 说明模型**不受控地**在几种写法之间摇摆。
  生成式做判定就是这样，你得把所有见过的写法都兜住。
- **`'Fal'` 这个键是 `max_new_tokens=1` 留下的化石。** 只生成一个 token 时，
  多字的 `False` 会被截成 `Fal`。键里有它就说明这个现象真实发生过。
- 归一化只做**一次**（`replace` 是整表替换），所以任何没被字典覆盖的取值会原样留在结果里。

### 4.3 批量推理版本（模板里改的）

notebook 是一行一行循环、一行一次前向，模板里改成了批处理：

```python
@torch.no_grad()
def predict_batch(prompts, batch=16):
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i+batch], return_tensors='pt', padding=True,
                  truncation=True, max_length=512).to(device)
        gen = model.generate(**enc, max_new_tokens=5, do_sample=False)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out

def parse(ans):
    a = str(ans).strip().lower()
    if 'true' in a or 'yes' in a:  return True
    if 'false' in a or 'no'  in a: return False
    return True                    # 兜底：多数类
```

三处改动，两处是改进、一处是隐患：

- `batch=16` + `padding=True`：一次前向处理 16 条，比逐条循环快得多。
- `max_new_tokens=5`：给足空间让模型把 `False` 说完，从根上消掉 `'Fal'` 这种半截词。
  这比在归一化字典里补键要正确得多。
- `truncation=True, max_length=512` —— **这里我留了个隐患。**
  prompt 的结构是「引导语 + 材料 + 问题」，**问题在最后**。
  按 512 截断，长材料会把问题整个切掉，模型就没得答了。
  notebook 里没截断，跑的时候控制台打了一条
  `Token indices sequence length is longer than the specified maximum sequence length (555 > 512)`
  ——说明确实有超过 512 的样本，但也说明它硬扛下来了、没报错。
  T5 用的是相对位置编码，超长只会让位置信息饱和，不会崩。
  **这一点在第 9 题的模板里是明确写过的**（那题的 prompt 同样是「文章在前、问题在后」），
  结果我在这题又踩回去了。模板里那条 `truncation=True` 应该改掉。

### 4.4 用训练集自检

```python
train_i = train[:300]                       # 只跑前 300 条，冒烟测试
...
acc = np.mean(np.array(preds) == y.values)
```

notebook 里存下来的自检结果是 **0.7533**（226 / 300）。
而模板注释里写了一句：这题的多数类基线大约是 **78%**。

**也就是说：零样本 flan-t5 在这 300 条上没打赢「全猜同一个答案」。**
这是个不太好受但必须写下来的结论。它解释了两件事：

1. 为什么这题真正的解法应该是**微调**（或者至少是「先用训练集选 prompt / 选模型」），
   而不是零样本硬上；
2. 为什么模板里最后要打印 `value_counts()` —— 如果输出几乎全是 `TRUE`，
   那就和「全预测一个值」没区别，而这是会被判零分的。

## 五、踩的坑

1. **编码。** 两个 CSV 都读不进来（`gb18030` 解码失败）。
   模板里的写法是 `pd.read_csv(path, encoding='utf-8', encoding_errors='replace')`
   —— 先按 UTF-8 读、遇到坏字节替换掉，先让数据进来再谈别的。
   卡片那边没有这层兜底，直接崩了，这也是为什么这题的数据字段表是空的。
2. **标签列名有两个版本。** notebook 用 `answer`，模板用 `Predicted_Results`。
   照抄模板会直接 `KeyError`。**「我以为是这个列名」是这类题最常见的翻车方式。**
3. **`max_new_tokens=1` 会切词。** `False` 变 `Fal`。
   修法不是往归一化字典里加键（那是打补丁），而是把 `max_new_tokens` 放大到 5。
4. **裸 `except` 吞异常。**
   ```python
   try:
       ...
       answers.append(answer[0])
   except:
       answers.append('-9999')      # 这个值后续映射不到任何东西
   ```
   单条出错时不中断整批，这个意图是对的；但 `'-9999'` 既不在归一化字典里，
   也不会被 `replace` 碰到，**它会原样写进提交文件，变成一个非法取值**。
   兜底值是合法值，这一条应该写进检查清单。
5. **`parse()` 的兜底是 `True`。**
   所有解析失败的样本都变成 `True`，也就是多数类。解析失败越多，输出越退化成单一值。
   必须配合输出分布自检（`value_counts()`）一起看。
6. **`train_i = train[:300]` 触发 SettingWithCopyWarning。**
   切片再赋值列，pandas 会警告「你可能在改一个视图」。这次是只读自检所以没出事，
   但如果那份 `train_i` 后面还要写出去，就得显式 `.copy()`。
7. **最后一步的文件名带个人信息。** 这类练习的提交约定是「姓名 + 证件号」的文件夹和文件名，
   练习时无所谓，但**这类文件绝对不能进仓库**。仓库里统一写成
   `考生姓名_results.csv` 这个占位形式。

## 六、小结

- **生成式做判定，代码确实短，但别把它当免费午餐。**
  这题零样本自检 0.7533，多数类基线约 0.78 —— 相当于什么都没做还倒亏一点。
  生成式的真正价值在于「没时间训练时能快速出一个合法结果」，
  而不是「比微调更好」。**先看清基线在哪，再决定要不要用零样本。**
- **生成式模型的输出必须当成脏字符串处理。** 归一化、兜底、输出分布自检，
  这三件一个都不能少。截断关键词（`Fal`）和裸 `except` 的 `-9999`
  都是「不报错但格式错」的典型。
- **prompt 里的顺序决定截断的后果。** 「材料在前、问题在后」时，
  标准的 `truncation=True, max_length=512` 会优先牺牲问题。
  这条教训我在第 9 题记过一次，这次又犯——所以它现在被写进了模板的注释里，
  而不是只留在复盘文章里。
- **素材不足时，短就是诚实。** 这题只有一个 notebook、两个 CSV 读不出来、
  评分口径也找不到。我把能确认的（提交格式、prompt 写法、自检数字、五个坑）写清楚，
  把不能确认的（真实字段名、测试集行数、评分指标）明确标成「不知道」。
  这比编一张看起来很完整的字段表有用得多。
