# 14 设备故障诊断

> 一句话总结：29 个类里 28 个类各只有 75 条样本，而 Macro-F1 把每个类平等看待——
> 这是我第一次真正理解"类别不平衡"和"用错评分指标"是两回事。

| | |
|---|---|
| 题型 | 表格 · 多分类（29 类 = 1 个正常 + 28 种故障） |
| 数据 | 训练 6,944 行 × 65 列（42 个小 Excel 拼起来：14 个 normal 文件 × 346 行 + 28 个 fault 文件 × 75 行）；测试 3,654 行 × 65 列（多一个 `ID`，没有标签） |
| 提交 | `考生姓名_results.csv`，列：`ID`, `Predicted_Results`（29 个类名之一，**大小写必须完全一致**） |
| 评分 | Macro-F1 |
| 代码 | [`code/14-tabular-multiclass`](../code/14-tabular-multiclass) · [`templates/14_表格多分类.py`](../templates/14_表格多分类.py) · [`templates/14_表格多分类_优化.py`](../templates/14_表格多分类_优化.py) |

## 一、题目在问什么

工业冷水机的状态诊断。传感器每时每刻在采 64 个字段——冷却水/冷冻水的进出口温度、
冷凝与蒸发侧的冷吨数、耗电功率 `kW`、能效比 `COP`、吸气/排气过热度、提升压力 `P_lift`、
电流百分比 `RLA%` 等等——要求判断设备当前状态是**正常**，还是 28 种故障状态中的哪一种。

故障一共 7 个家族，每个家族 4 个严重程度：

```text
cf1..cf4    eo1..eo4    fwc1..fwc4    fwe1..fwe4
nc1..nc4    rl1..rl4    ro1..ro4
```

`cf1` 到 `cf4` 表示同一类故障的 4 个等级。所以这是个 **29 类**的分类问题
（28 种故障 + `normal`）。

**数据组织方式是这题最大的特点**：训练数据不是一个文件，是 **42 个小 Excel**，
按 `normal/` 和 `fault/` 两个目录分开放；测试是一个文件。

## 二、数据长什么样

先把 43 个文件的样子列出来（来自字段卡）：

| 目录 / 文件 | 行 × 列 | 说明 |
|---|---|---|
| `Training_Data/normal/` 下 14 个文件 | 346 × 65 | 每个文件 346 行，标签全是 `normal`，合计 4,844 行 |
| `Training_Data/fault/` 下 28 个文件 | 75 × 65 | 每个文件对应一种故障，合计 2,100 行 |
| `test_data_all.xlsx` | 3,654 × 65 | 比训练集多一个 `ID`，没有标签 |

也就是说：正常样本 4,844 条，**每一种故障只有 75 条**。

65 列 = 64 个传感器字段 + `Class`（训练）/ 64 个字段 + `ID`（测试）。
字段太多，挑前 40 个列成表，剩下的 25 个只列名字：

| 列名 | 类型 | unique | 说明 |
|---|---|---|---|
| `TEI` / `TWEI` | float64 | 510 / 163 | 蒸发器进水温度（测量值 / 平滑值） |
| `TEO` / `TWEO` | float64 | 412 / 162 | 蒸发器出水温度 |
| `TCI` / `TWCI` | float64 | 599 / 193 | 冷凝器进水温度 |
| `TCO` / `TWCO` | float64 | 787 / 242 | 冷凝器出水温度 |
| `TSI` / `TSO` | float64 | 749 / 628 | 蒸汽侧进出温度 |
| `TBI` / `TBO` | float64 | 496 / 511 | 换热器/储液罐侧进出温度 |
| `Cond Tons` | float64 | 2507 | 冷凝侧冷吨数（功率量） |
| `Cooling Tons` | float64 | 1750 | 制冷量 |
| `Shared Cond Tons` | float64 | 2354 | 分摊到本机组的冷凝吨数 |
| `Cond Energy Balance` | float64 | 2447 | 冷凝侧能量平衡（负值居多） |
| `Evap Tons` | float64 | 2669 | 蒸发侧冷吨数 |
| `Shared Evap Tons` | float64 | 2666 | 分摊到本机组的蒸发吨数 |
| `Building Tons` | float64 | 1540 | 建筑侧负荷 |
| `Evap Energy Balance` | float64 | 879 | 蒸发侧能量平衡 |
| `kW` | float64 | 2442 | 实时功率 |
| `COP` | float64 | 2007 | 能效比（性能下降时降低） |
| `kW/Ton` | float64 | 2009 | 单位制冷量耗电 |
| `FWC` / `FWE` | float64 | 453 / 186 | 冷却水/冷冻水流量 |
| `TEA` / `TCA` | float64 | 108 / 157 | 蒸发器/冷凝器趋近温度 |
| `TRE` / `PRE` | float64 | 144 / 149 | 蒸发/冷凝压力 |
| `TRC` / `PRC` | float64 | 164 / 167 | 冷凝器饱和温度/压力 |
| `TRC_sub` | float64 | 193 | 过冷度 |
| `T_suc` / `Tsh_suc` | float64 | 183 / 140 | 吸气温度 / 吸气过热度 |
| `TR_dis` / `Tsh_dis` | float64 | 249 / 507 | 排气温度 / 排气过热度 |
| `P_lift` | float64 | 650 | 提升压力 |
| `Amps` / `RLA%` | int64 | 70 / 70 | 电流 / 额定负载电流百分比 |
| 其余 25 列 | — | — | `Heat Balance (kW)`, `Heat Balance%`, `Tolerance%`, `Unit Status`, `Active Fault`, `TO_sump`, `TO_feed`, `PO_feed`, `PO_net`, `TWCD`, `TWED`, `VSS`, `VSL`, `VH`, `VM`, `VC`, `VE`, `VW`, `TWI`, `TWO`, `THI`, `THO`, `FWW`, `FWH`, `FWB` |

> 字段卡是按列截断的（列出前 40 列），后面 25 列只留了表头名字，
> 所以它们的 unique / 样例我**没有**，不能凭空补。

三个从字段层面就该注意的点：

1. **同一个含义的字段在不同文件里写法不一致。** 42 个训练文件里，
   绝大多数把能量平衡写成 `Heat Balance (kW)`，但有 **4 个文件写成了 `Heat Balance`**
   （我抽到的卡片里是 `fwc3` / `fwc4` / `normal3` / `normal9`）。
   这会让"按列名拼数据"直接炸掉。
2. **`Unit Status` / `Active Fault` 很像标签本身。** 在我抽到的样例行里
   `Unit Status` 恒为 20、`Active Fault` 恒为 0。要么是常量，要么是另一个维度的状态编码。
   notebook 没有特意处理它们。
3. **`Class` 大小写**：训练集里目标列叫 `Class`（首字母大写），不是 `class`。

## 三、思路

1. **先把 42 个文件拼起来**：遍历 `normal/` 和 `fault/` 两个目录，逐个 `read_excel` 后 `concat`。
2. **对齐特征列**：训练集和测试集的列做一个差集检查，只保留两边都有的列
   （训练集有 `Class`，测试集有 `ID`，另外就是那个名字不一致的 `Heat Balance`）。
3. **预处理**：中位数填缺失 → `VarianceThreshold(0.01)` 过滤低方差列 → `StandardScaler` 标准化。
4. **编码目标**：`LabelEncoder` 把 29 个类名变成 0~28。
5. **多模型 5 折对比**：XGBoost / RandomForest / LightGBM / GradientBoosting / ExtraTrees，
   同时看 Macro-F1 和 Micro-F1。
6. **用 LightGBM 调参**：随机搜索 15 组参数，8 折交叉验证 + 早停，
   记录每组的平均 Macro-F1 和平均最佳轮数。
7. **全量重训，`inverse_transform` 反编码回类名，写文件。**

## 四、关键代码

**① 42 个 Excel 拼成一张表**：

```python
norm  = '.../Training_Data/normal/'
fault = '.../Training_Data/fault/'

df_norms = pd.DataFrame()
for i in os.listdir(norm):
    df_norms = pd.concat([df_norms, pd.read_excel(norm + i)])

df_faults = pd.DataFrame()
for i in os.listdir(fault):
    df_faults = pd.concat([df_faults, pd.read_excel(fault + i)])

df_train = pd.concat([df_norms, df_faults]).reset_index(drop=True)
```

这段现在是模板里 [`14_表格多分类.py`](../templates/14_表格多分类.py) 的 `load_chiller()` 函数——
**"一个题目拆成几十个文件"这种形态我只在这题遇到过**，所以专门包成了一个函数。

**② 用集合差集对齐两边列名**（这题最该先跑的一段）：

```python
set(df_train.columns) - set(test_df.columns)   # -> {'Class', 'Heat Balance'}
set(test_df.columns) - set(df_train.columns)   # -> {'ID'}
df_train = df_train.drop('Heat Balance', axis=1)
```

`Class` 和 `ID` 是预期内的，`Heat Balance` 是意外——
**跑一次差集只需要一行代码，却能省掉几十分钟排查"为什么 concat 之后多出一列全 NaN"**。
模板里最后落成的写法是正向的：

```python
feat = [c for c in train.columns if c in test.columns and c not in (TARGET, ID_COL)]
```

**③ 反编码，保证类名大小写一字不差**：

```python
le = LabelEncoder()
y_train = le.fit_transform(df_train['Class'].astype(str))

y_test_pre = le.inverse_transform(best.predict(x_test))
```

题目明确说了 `Predicted_Results` 必须是 `normal` / `cf1` / … / `ro4` 这些字样之一，
**大小写不同就不给分**。手写映射表迟早会写错一个，
用 `inverse_transform` 从编码器里还原是最保险的做法。

**④ 方差过滤 + 标准化**：

```python
vt = VarianceThreshold(threshold=0.01)
x_train = vt.fit_transform(x_train)
x_test  = vt.transform(x_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)
```

`VarianceThreshold` 顺手把常量列（比如前面提到的 `Unit Status` / `Active Fault` 如果真是常量）
清掉了，不用我手动挑。

**⑤ 调参时同时记 Macro-F1 和最佳轮数**：

```python
for i in range(N_ITER_SEARCH):           # N_ITER_SEARCH = 15
    current_params = {k: (np.random.choice(v) if isinstance(v, list) else v.rvs())
                      for k, v in param_dist.items()}
    current_params.update({'n_estimators': 5000, 'objective': 'multiclass',
                           'num_class': num_classes, 'class_weight': 'balanced', ...})
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_train1, y_train1)):
        model = lgb.LGBMClassifier(**current_params)
        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        cv_macro.append(f1_score(y_fold_val, model.predict(X_fold_val), average='macro'))
        cv_iterations.append(model.best_iteration_)
```

`param_dist` 用的是 `scipy.stats` 的分布对象（`uniform` / `randint`），
`value.rvs()` 抽一个样本——这是一次性搜多组参数的省事写法。
**同时记 `best_iteration_` 是关键**：搜完可以用"各折平均最佳轮数"当作全量重训的
`n_estimators`，比自己猜一个数靠谱。

**⑥ 优化版：样本权重 + 软投票**，这就是 [`14_表格多分类_优化.py`](../templates/14_表格多分类_优化.py) 加的三件事：

```python
from sklearn.utils.class_weight import compute_sample_weight
sw = compute_sample_weight('balanced', y)      # 每个类等权
...
# 软投票:只用 LGBM+XGB 各半（实测 RF 太弱,加进去反而拖后腿）
ens = 0.5 * probas['LGBM'] + 0.5 * probas['XGB']
cv_acc['ENSEMBLE(LGBM+XGB)'].append(f1_score(y[va], ens.argmax(1), average='macro'))
```

那个文件的文件头把改动记得很清楚：**给 XGB 也加上类别平衡之后，Macro-F1 从 0.7665 变成 0.7971**；
RF 因为太弱被踢出集成。**只有被实测验证过的改动才留在模板里**，这是我现在写模板的习惯。

## 五、踩的坑

**坑 1：列名不一致。** 前面说的 `Heat Balance` 和 `Heat Balance (kW)`——
4 个文件和其他 38 个写法不同。如果直接 `pd.concat`，pandas 不会报错，
它会给你**两列**，其中一列有 4 个 346 行有值、其余全 NaN。
这种"静默多出一列"的 bug 在 65 列里非常难肉眼发现。

**坑 2：标准化在切折之前做了。** notebook 的顺序是：
先 `StandardScaler` 在全量训练集上 `fit_transform`，然后才 `StratifiedKFold` 切 5 折。
这意味着**每一折的验证集统计量（均值/标准差）已经参与了缩放**，
交叉验证分数会略微偏乐观。正确顺序应该是"在每一折内部 `fit` 缩放器"。
这题特征量纲差异大（`Amps` 几十、`Cond Tons` 几百、`kW/Ton` 不到 1），
所以缩放本身是必要的，只是位置放错了。

**坑 3：最后用来预测的不是 `final_model`。** 这是复查时发现的最大的一个坑。
notebook 花了 15 组随机搜索 + 8 折交叉验证，挑出最优参数，
然后在 `final_model` 上用全量数据重训了一遍：

```python
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(x_train, y_train)
```

但紧接着的预测用的是**上一段 5 折循环里留下的 `model` 变量**：

```python
y_test_pre = model.predict(x_test)      # ← 不是 final_model
```

也就是说：**搜出来的那组参数、那次全量重训，最后完全没被用上。**
再往细看更离谱：`models` 这个字典的顺序是 XGBoost → RF → LGBM → GB → ExtraTrees，
而循环体里 `model = model.__class__(**model.get_params())` 会把 `model` 这个名字重新绑定，
所以**循环结束后 `model` 指向的是 ExtraTrees 在最后一折上训出来的那个实例**——
也就是 5 折对比里 Macro-F1 最低的那个模型（`0.4501`）。
这个坑非常隐蔽：代码能跑、结果也不荒谬（预测分布看起来很正常），
只是"你以为在交 LGBM，实际在交 ExtraTrees"。
现在我的模板里会把最终模型统一叫 `final_model`，并且预测那一行**紧挨着**训练那一行写。

**坑 4：写出去的列名不对。** notebook 最后是：

```python
result = test_df[['ID']]
result['Class'] = y_test_pre
```

列名是 `ID` / `Class`，而题目要的是 `ID` / `Predicted_Results`。
和坑 3 一样，这属于"能跑通但会 0 分"的问题——**提交格式必须在最后加断言自检**，
这一条后来成了 16 那套模板里的固定动作。

**坑 5：搜索记录只打出来 12 组。** 代码写的是 15 组，
但 notebook 的输出在第 12 组之后就被截断了，所以第 13~15 组试了什么参数我**没有记录**。
`best_result = max(results, key=lambda x: x['mean_macrof1'])` 给出的最优是
**5 折平均 Macro-F1 `0.81060`**，比打印出来的 12 组里最高的 `0.80925` 还高，
说明后 3 组里有更好的，但具体是什么已经查不到了。
这个数字是**搜索阶段的交叉验证记录**，而且因为坑 3，
**它和最终交出去的那份结果没有任何关系**——不能把它当成成绩。

**坑 6：Macro-F1 下不能用准确率思考。** 正常样本 4,844 条、
每种故障只有 75 条，一共 6,944 条。如果模型把大多数样本判成"正常几类"，
Micro-F1（≈准确率）能很高，但 Macro-F1 会因为 28 个故障类几乎全错而崩掉。
5 折对比的记录里这个落差非常明显：RandomForest 的
Macro-F1 `0.4614` 对 Micro-F1 `0.2049`，ExtraTrees 是 `0.4501` / `0.2630`——
两个数字完全对不上，**说明它们在个别类上还不错、但整体覆盖率很差**。
反观 LightGBM 是 `0.7948` / `0.9183`，两边都高，这才是能用的模型。
`class_weight='balanced'` / `compute_sample_weight('balanced')` 在这一题不是可选项，是必需品。

## 六、小结

- 多文件训练集的第一件事是**拼表 + 对齐列名**，而且必须用
  `set(train.columns) - set(test.columns)` 真的跑一遍，不能靠眼睛看。
- **多分类的评分指标一定要先看清是 Macro 还是 Micro。**
  Macro 对稀有类等权，这题的 28 个 75 条样本的类和 4,844 条的 `normal` 一样重要；
  这也直接决定了必须做类别平衡。
- **`inverse_transform` 反编码** 比手写类名映射安全，尤其是题目要求大小写一致的时候。
- 这题留下的两个教训（"搜完参要用 `final_model`"和"提交列名要断言自检"）
  后来都写进了 [`templates/16_结构化实战模板.py`](../templates/16_结构化实战模板.py) 里。
- 模板层面，这题沉淀出两个文件：基础版 [`14_表格多分类.py`](../templates/14_表格多分类.py)
  负责"拼表 → 对齐列 → 编码 → 多模型 → 反编码"这条主线；
  优化版 [`14_表格多分类_优化.py`](../templates/14_表格多分类_优化.py)
  加了样本权重、LGBM 网格调参和 LGBM+XGB 软投票，并且把"RF 拖后腿"这个结论写在了文件头。
