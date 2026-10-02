# 数据分析练习笔记

这是一份**个人练习记录**。从 2021 年到 2026 年，我陆续做了十几个结构化的数据分析小项目，
每个项目都是同一套流程：拿到数据 → 判题型 → 套模板 → 本地验证 → 交一份结果文件。

做到后面我发现自己一直在重复同样的动作，于是把可复用的部分抽成了模板库，
每个项目再写一篇复盘。**这个仓库就是那套模板 + 那些复盘。**

> 一句话原则：**没有输出文件 = 0 分。**
> 宁可先跑出一个保底结果，再回头优化，绝不要卡在"想清楚"上拖到超时。

---

## 目录

| # | 主题 | 题型 | 复盘 | 代码 |
|---|---|---|---|---|
| 01 | 影评情感分析 | 文本 · 二分类 | [文章](posts/01-sentiment.md) | [`code/01-sentiment`](code/01-sentiment) |
| 02 | 恶意评论识别与分级 | 文本 · 多标签 | [文章](posts/02-toxic-multilabel.md) | [`code/02-toxic-multilabel`](code/02-toxic-multilabel) |
| 03 | 用户电影评分预测 | 回归 | [文章](posts/03-movie-rating.md) | [`code/03-movie-rating`](code/03-movie-rating) |
| 04 | 住宿日租价格预测 | 回归 | [文章](posts/04-rental-price.md) | [`code/04-rental-price`](code/04-rental-price) |
| 05 | 中英商品标题分类 | 文本 · 多分类 | [文章](posts/05-product-category.md) | [`code/05-product-category`](code/05-product-category) |
| 06 | 相似项检索 | 检索 / 排序 | [文章](posts/06-similar-item.md) | [`code/06-similar-item`](code/06-similar-item) |
| 07 | AI 生成文本识别 | 文本 · 二分类 | [文章](posts/07-ai-text-detect.md) | [`code/07-ai-text-detect`](code/07-ai-text-detect) |
| 08 | 文本对相似度 | 文本 · 二分类 | [文章](posts/08-text-similarity.md) | [`templates/08_文本对相似度.py`](templates/08_文本对相似度.py) |
| 09 | 阅读理解多选 | 文本 · 多分类 | [文章](posts/09-reading-qa.md) | [`code/09-reading-qa`](code/09-reading-qa) |
| 10 | 多表关联 + 欺诈识别 | 表格 · 二分类 | [文章](posts/10-multitable-join.md) | [`code/10-multitable-join`](code/10-multitable-join) |
| 11 | SQL 取数 + 多标签 | 文本 · 多标签 | [文章](posts/11-sql-multilabel.md) | [`code/11-sql-multilabel`](code/11-sql-multilabel) |
| 12 | 文本回归 | 回归 | [文章](posts/12-text-regression.md) | [`code/12-text-regression`](code/12-text-regression) |
| 13 | 问答判定 | 生成式 / 判定 | [文章](posts/13-qa-judgement.md) | [`code/13-qa-judgement`](code/13-qa-judgement) |
| 14 | 设备故障诊断 | 表格 · 多分类 | [文章](posts/14-tabular-multiclass.md) | [`code/14-tabular-multiclass`](code/14-tabular-multiclass) |
| 15 | 客户流失预测 | 表格 · 二分类 | [文章](posts/15-churn.md) | [`code/15-churn`](code/15-churn) |
| 16 | 电商用户行为预测 | 表格 · 二分类 | [文章](posts/16-ecommerce.md) | [`code/16-ecommerce`](code/16-ecommerce) |

---

## 方法论

| 文档 | 内容 |
|---|---|
| [00-解题流程](docs/00-解题流程.md) | 遇到没练过的题照着走：读题面 → 判题型 → 套模板 → 本地验证 → 交卷 |
| [01-实战速查](docs/01-实战速查.md) | 时间分配、检查清单、最容易丢分的细节 |
| [02-AI协作模板](docs/02-AI协作模板.md) | 怎么把 AI 当副驾用：喂结构化摘要而不是原始数据 |

### 核心心法

**喂「结构化摘要」，别喂原始数据。** 几十万行的表格是几百万 token，但一张 10 行的字段摘要
才几百 token。上下文小不是瓶颈，喂错了东西才是。先自己用 pandas 扫一眼，压成摘要再往下走：

```
【任务】<一句话>
【训练】<N 行 × M 列>：
  <列名> <类型> (<unique 数> unique)   例: <2 条样例>
  ...
  <目标列> <类型> (目标)
【测试】<M 行，字段同上/无目标列>
【提交】<文件名>，<几列>：<列名1>, <列名2>, <提交列>（<值类型/范围/整数约束>）
【评分】<RMSE / Macro-F1 / accuracy ...>
```

这张卡片同时也是**你检查自己有没有理解错题**的最好工具——填不出来的格子就是没读懂的地方。

---

## 模板库

`templates/` 里的文件是这几年攒下来的通用骨架，按练习顺序编号。
同一个题型在不同年份写过好几版（比如 `02_多标签分类*.py` 有三版），
**留在仓库里是为了看到演进过程**，不是为了让你每个都用。

| 文件 | 用途 |
|---|---|
| [`01_文本二分类.py`](templates/01_文本二分类.py) | TF-IDF + 线性模型，最基础的文本二分类 |
| [`02_多标签分类.py`](templates/02_多标签分类.py) / [`_改进版`](templates/02_多标签分类_改进版.py) / [`_通用版`](templates/02_多标签分类_通用版.py) | 多标签，含 OneVsRest 与阈值选择 |
| [`02_阈值调优实验.py`](templates/02_阈值调优实验.py) | 逐标签调阈值的实验脚本 |
| [`04_回归_改进版.py`](templates/04_回归_改进版.py) | 回归任务骨架 |
| [`05_中英多分类.py`](templates/05_中英多分类.py) | 中英混排文本的分类 |
| [`06_相似项检索.py`](templates/06_相似项检索.py) | 向量化 + 相似度检索 |
| [`07_文本二分类.py`](templates/07_文本二分类.py) | 二分类（第三版） |
| [`08_文本对相似度.py`](templates/08_文本对相似度.py) | 句对相似度，含 embedding 缓存 |
| [`09_文本多选QA.py`](templates/09_文本多选QA.py) | 多选式阅读理解 |
| [`10_表格二分类.py`](templates/10_表格二分类.py) / [`10_多表join_特征工程.py`](templates/10_多表join_特征工程.py) | 表格二分类 + 多表聚合 |
| [`11_多标签分类.py`](templates/11_多标签分类.py) / [`11_多标签分类_实战.py`](templates/11_多标签分类_实战.py) / [`11_特殊数据加载.py`](templates/11_特殊数据加载.py) | 逗号分隔标签的处理与 SQL 取数 |
| [`13_flan_t5生成式.py`](templates/13_flan_t5生成式.py) | 生成式做判定 |
| [`14_表格多分类.py`](templates/14_表格多分类.py) / [`_优化`](templates/14_表格多分类_优化.py) | 表格多分类 |
| [`15_表格二分类.py`](templates/15_表格二分类.py) | 表格二分类 |
| [`16_结构化实战模板.py`](templates/16_结构化实战模板.py) | **最终版**，把前面所有教训合并成一个可配置骨架 |
| [`16_速通版.py`](templates/16_速通版.py) | 最小可用版，时间不够时用 |
| [`16_特征工程片段库.py`](templates/16_特征工程片段库.py) / [`16_自动检测特征工程.py`](templates/16_自动检测特征工程.py) / [`16_特征工程已填好.py`](templates/16_特征工程已填好.py) | 特征工程片段、自动检测该做什么特征 |

---

## 关于数据

**仓库里没有数据文件**，只有代码。原因很直接：原始数据加起来 400+ MB
（其中一个 SQLite 库 200 MB，两个 word2vec 权重各 90 MB），塞进 git 没有意义。

每篇复盘里都写了该练习的数据长什么样（行数、列名、类型、目标列），
照着字段描述自己准备一份同构的数据就能把代码跑起来。

---

## 仓库结构

```
.
├── README.md            本文件
├── posts/               每个练习一篇复盘
├── code/                每个练习自己的 notebook 和脚本
├── templates/           通用模板库
└── docs/                方法论
```

这个仓库是从我本地的练习目录里**挑出来**的，不是整个目录直接推上去的。

排除掉的东西分三类：

1. **数据**。原始数据 400+ MB（其中一个 SQLite 库 200 MB，两个 word2vec 权重各 90 MB），
   放进 git 没有意义。
2. **别人的东西**。原始目录里有题目原文文档和配套数据，那些不是我写的，不往这里放。
   复盘里的题目描述都是我自己的话。
3. **不该公开的**。临时目录、`*.bak` 快照、提交产物，以及**文件名里带个人信息的文件**。

整理时还做了一件事：把 notebook 里指向数据的**本机绝对路径**改成了 `./data/...` 这种相对路径，
顺手把路径里的 Windows 用户名换成了 `user`。现在 clone 下来，把 data 放到 `code/<练习>/data/`
下面就能跑。少数几条出现在报错输出里的临时目录路径（`C:/Users/user/AppData/...`）没动，
那是 Jupyter 自己打的，不影响运行。

