# -*- coding: utf-8 -*-
# ============================================================================
# 13_特征工程片段库_202609.py —— 「看列选招式」的零件库（复制到 12 模板里用）
# ============================================================================
# ---------------------------------------------------------------------------
# 【怎么用，三步】
#   1) 先跑 12_模板_202609结构化实战模板.py，跑到「★ 特征片段粘贴处」
#      （那一节已经准备好了 full / num_cols / cat_cols / new_feats / keep_feats）
#   2) 回到本文件，先看【第 0 节 速查表】确认你的列属于哪一类，
#      再跳到对应片段，整段复制粘贴到模板的粘贴处
#   3) 每个片段都是自包含的，贴 2~5 个就够，不用全贴
#
# 【选招式的核心思路 —— 只看三件事】
#   · 这列是"计数 / 金额 / 时长 / 比率"里的哪一种？ -> 决定能不能做比率和差值
#   · 这列有没有"业务顺序"？ -> 有顺序用顺序编码；没顺序用类别编码
#   · 这列取值多不多？ -> 少：独热/交叉；中等：目标编码；很多：频次编码
# ============================================================================


# ============================================================================
# 第 0 节  速查表：你的列长什么样 -> 用哪个片段
# ============================================================================
# +------------------------+--------------------------------------------+--------+
# | 列的"长相"（看名字+看值）| 代表字段（历年真题）                       | 用片段 |
# +------------------------+--------------------------------------------+--------+
# | 次数 / 个数            | Administrative, ProductRelated, quantity,   | ①②③   |
# |                        | 登录次数、购买笔数、点击量                   |        |
# | 金额 / 价格            | MonthlyCharges, TotalCharges, amount,       | ①③⑨   |
# |                        | 订单金额、客单价                            |        |
# | 时长 / 时间长度        | Admin_Duration, Product_Duration, tenure,   | ①②③   |
# |                        | 通话时长、停留秒数                          |        |
# | 比率类（天然 0~1）     | BounceRates, ExitRates, 转化率, 逾期率      | ③⑥     |
# | 等级 / 程度            | SpecialDay, 满意度1-5, 信用等级A-E          | ③⑥     |
# | 有顺序的类别           | Contract(月/1年/2年), Month, 学历, 套餐档位 | ⑤      |
# | 无顺序类别（少，≤10）  | gender, Partner, Weekend(是/否)             | ⑦⑨     |
# | 无顺序类别（中，10~500）| PaymentMethod, InternetService, 城市, 职业  | ③④⑦   |
# | 无顺序类别（很多）     | CustomerID, 邮编, 商品码, 设备号            | ④③     |
# | 日期 / 时间戳          | 注册日期, 下单时间, 账单月                  | ⑧      |
# | 短文本（名称/备注）    | remark, 商品标题, 备注说明                  | ⑪      |
# | 纯序号（全唯一）       | ID, 流水号                                  | 删掉！ |
# +------------------------+--------------------------------------------+--------+
#
# 片段编号速记：①比率差值 ②行统计 ③目标编码 ④频次编码 ⑤有序映射 ⑥分箱
#              ⑦类别交叉 ⑧日期特征 ⑨log变换 ⑩异常打分 ⑪文本特征
#
# 性价比排序（只贴 2 个就贴 ①②；贴 4 个再加 ③⑤）
#   ①比率差值 > ②行统计 > ③OOF目标编码 > ⑤有序映射 > ④频次编码
#   > ⑧日期 > ⑥分箱 > ⑦类别交叉 > ⑨log > ⑪文本 > ⑩异常打分


# ============================================================================
# 第 0.5 节  字段自动分型（不确定哪列属于哪类？先跑这一段看输出）
# ============================================================================
# 这一段不用粘贴进模板，只是帮你把列归类。粘到模板 cell 3 后面跑一次即可。

def _guess_col_type(name, s):
    """按列名关键词 + 数据特征猜语义类型，返回 (中文类型, 推荐片段)。"""
    n = str(name).lower()
    uniq = s.nunique(dropna=True)
    if uniq <= 2:
        return '二值(是/否)', '⑦⑨'
    if ('duration' in n or 'tenure' in n or '时长' in n or '停留' in n):
        return '时长(数值)', '①②③'
    if any(k in n for k in ['date', 'time', '日期', '时间']):
        return '日期/时间戳', '⑧'
    if any(k in n for k in ['amount', 'charge', 'price', 'fee', 'cost', 'revenue',
                            '金额', '价格', '费用', '客单']):
        return '金额(数值)', '①③⑨'
    if any(k in n for k in ['rate', 'ratio', 'pct', 'percent', '率', '占比',
                            'bounce', 'exit']):
        return '比率(0~1)', '③⑥'
    if any(k in n for k in ['num', 'cnt', 'count', 'qty', 'quantity', 'visit',
                            'times', '次数', '数量', '笔数', '个数']):
        return '计数(数值)', '①②③'
    if any(k in n for k in ['id', 'code', '编号', '流水']):
        return '疑似ID/编码', '④③ (或删掉)'
    if s.dtype == object:
        if uniq <= 10:
            return '类别(低基数)', '⑦⑨'
        if uniq <= 500:
            return '类别(中基数)', '③④⑦'
        return '高基数/文本', '④③⑪'
    if uniq <= 10:
        return '数值(取值少,像等级)', '③⑥'
    return '连续数值', '①②⑨'


try:
    P('\n===== 字段分型建议（规则猜测，请自己确认）=====')
    _rows = []
    for _c in list(num_cols) + list(cat_cols):
        _t, _seg = _guess_col_type(_c, full[_c])
        _rows.append({'字段': _c, '取值数': full[_c].nunique(dropna=True),
                      '类型判断': _t, '推荐片段': _seg,
                      '样例': str(full[_c].dropna().unique()[:3])[:34]})
    P(pd.DataFrame(_rows).to_string(index=False))
except NameError:
    pass          # 单独运行本文件时跳过（本文件是片段库，不是可执行脚本）


# ============================================================================
# 片段 ① 比率 / 差值  —— 性价比最高，几乎每题都能用
# ============================================================================
# 【什么列用】成对的"总量"类字段，典型组合：
#     · 次数 vs 次数   ProductRelated / Administrative        -> 浏览结构占比
#     · 时长 vs 时长   Product_Duration / Admin_Duration      -> 兴趣集中度
#     · 次数 vs 时长   ProductRelated / Product_Duration      -> 平均每页停留
#     · 金额 vs 月数   TotalCharges / tenure                  -> 月均消费
#     · 金额 vs 金额   TotalCharges / MonthlyCharges          -> 已用月数（反推）
# 【什么列别用】意思不相关的两列相除（如 年龄/地区码）-> 纯噪声
#   特别注意：二值标记列（Weekend、是否类）和类别编码列（OperatingSystems、
#   Browser、Region、TrafficType）【不能】拿来做比率 —— 代码已自动过滤掉它们
# 【插到哪】模板「★ 特征片段粘贴处」
# 【注意】
#   · 分母 +1e-6 防除零；分母为 0 的位置用中位数兜底
#   · 比率必须缩尾，否则极端比率会把树带偏
#   · 字段多别全组合：先用互信息挑前 PAIR_TOP_N 个（20 个 -> 190 对）
# ----------------------------------------------------------------------------
PAIR_TOP_N = 15            # 只用互信息最强的前 N 个数值字段做两两衍生

# ⚠ 关键过滤（实测踩过）：只按"互信息高"挑字段会挑出二值标记列和类别编码列，
#   于是造出 Weekend/Informational、Weekend-SpecialDay 这种毫无业务含义的噪声。
#   比率和差值只对"有量纲的量"（次数/金额/时长）有意义，所以先筛一遍：
#     · 排除二值列（取值≤2，如 Weekend、是否类标记）
#     · 排除疑似类别编码列（OperatingSystems / Browser / Region / TrafficType…）
#     · 排除取值很少的列（≤5 个取值，本身像等级而不是量）
_CODE_WORDS = ['os', 'browser', 'region', 'traffic', 'type', 'code', 'id', 'flag',
               'is_', '编码', '代码', '类型', '编号', '地区', '系统']


def _is_quantity(col):
    """判断一列是不是"可做比率/差值的量"（而不是标记位或类别编码）。"""
    s = full[col].dropna()
    if len(s) == 0:
        return False
    _u = s.nunique()
    # ① 取值太少（≤5）-> 是标记/等级，不是量
    if _u <= 5:
        return False
    # ② 名字像类别编码（OperatingSystems/Browser/Region/TrafficType/各种 code）
    n = str(col).lower()
    if any(w in n for w in _CODE_WORDS):
        return False
    # ③ 看起来是"整数码"：取值都是整数、种类有限、而且种类数/最大值的比例很高
    #    例子：OperatingSystems 取值 {1..8} -> 8 种/最大值 8 = 1.0 -> 是编码
    #          数量类字段取值可能到几千 -> 比例很低 -> 是量
    try:
        v = s.values.astype(float)
        if np.allclose(v, np.round(v)) and v.min() >= 0 and _u <= 60:
            rng = v.max() - v.min()
            if rng > 0 and (_u / (rng + 1)) > 0.5:
                return False
    except Exception:
        pass
    return True


_num_all = [c for c in num_cols if _is_quantity(c)]
if len(_num_all) < 3:                          # 万一全被筛掉，退回原始清单
    _num_all = list(num_cols)
    P('  ⚠ 可用"量"字段太少，退回全部数值列')
else:
    P('\n片段① 过滤掉的非量纲列（不参与比率差值）:',
      [c for c in num_cols if c not in _num_all])

_y_tr = full.loc[full['__is_train'] == 1, TARGET].astype(int).values
from sklearn.feature_selection import mutual_info_classif as _mi_clf
_mi = pd.Series(
    _mi_clf(full.loc[full['__is_train'] == 1, _num_all].values.astype('float32'),
            _y_tr, random_state=42, n_neighbors=3),
    index=_num_all).sort_values(ascending=False)
_pair_base = _mi.head(PAIR_TOP_N).index.tolist()
P('片段① 参与两两衍生的字段（按互信息）:', _pair_base)

from itertools import combinations as _comb1          # 片段① 自己 import，不依赖别的片段
_n_new = 0
for _a, _b in _comb1(_pair_base, 2):
    _va, _vb = full[_a].values.astype(float), full[_b].values.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        _r = _va / (_vb + 1e-6)
    _r = np.where(np.isfinite(_r), _r, np.nan)
    if np.isnan(_r).mean() > 0.5:
        continue
    _cn = '%s__div__%s' % (_a, _b)
    full[_cn] = np.where(np.isnan(_r), np.nanmedian(_r), _r)
    full[_cn] = full[_cn].clip(*np.nanpercentile(_r, [0.1, 99.9]))     # 比率缩尾
    new_feats.append(_cn)
    _n_new += 1
    _sn = '%s__sub__%s' % (_a, _b)
    full[_sn] = _va - _vb
    new_feats.append(_sn)
    _n_new += 1
P('片段① 完成：%d 对 -> 新增 %d 列比率/差值' % (_n_new // 2, _n_new))

# 【手写版，更精准】知道该做哪几对时，直接写死（推荐）
PAIRS = [
    ('Product_Duration', 'Admin_Duration'),   # 产品页 vs 行政页时长
    ('ProductRelated', 'Administrative'),     # 产品页 vs 行政页次数
    ('TotalCharges', 'MonthlyCharges'),       # 已用月数
    ('MonthlyCharges', 'tenure'),             # 月均消费
]
for a, b in PAIRS:
    if a in full.columns and b in full.columns:
        full['%s__div__%s' % (a, b)] = full[a] / (full[b] + 1e-6)
        new_feats.append('%s__div__%s' % (a, b))


# ============================================================================
# 片段 ② 行统计（横向聚合）—— 一行里有很多"同类"字段时最有效
# ============================================================================
# 【什么列用】一组语义相同的列：
#     · 各类页面数：Administrative / Informational / ProductRelated
#     · 各类时长：  Admin_Duration / Info_Duration / Product_Duration
#     · 各项费用：  月租 + 流量费 + 增值费
#     · 各科成绩：  语文 / 数学 / 英语
# 【原理】"各项之间的结构"（最大项占比、离散程度）往往比单列更有区分度
# 【插到哪】模板「★ 特征片段粘贴处」
# 【怎么选列】把同类的列填进 GROUPS —— 强烈建议填，别留空！
#   留空时的自动分组只按"数量级"粗略分（下面有兜底），
#   不同量纲的列（跳出率 0.02 和 时长 553 秒）混在一组算出来的和/均值是噪声
# ----------------------------------------------------------------------------
GROUPS = {
    # 组名: [该组的列]      ← 每组必须是"语义同类 + 量纲相近"的列
    #
    # 【电商数据（2026.06 那种）—— 直接照抄】
    # '页面数': ['Administrative', 'Informational', 'ProductRelated'],
    # '时长':   ['Admin_Duration', 'Info_Duration', 'Product_Duration'],
    #
    # 【电信流失数据（2026.03 那种）—— 直接照抄】
    # '附加服务数': ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    #                'TechSupport', 'StreamingTV', 'StreamingMovies'],
    # '费用':       ['MonthlyCharges', 'TotalCharges'],
    #
    # 【通用判断法】问自己两个问题：
    #   ① 这几列"数的是同一类东西"吗？（都是页面数 / 都是时长 / 都是费用）
    #   ② 它们的量纲接近吗？（别把 0.02 的比率和 553 的时长放一组）
    #   两个都答"是"才放一组。
}

if not GROUPS:
    # 兜底：按"数量级"自动分组，避免把跳出率(0.02)和时长(553秒)混在一起
    #   ⚠ 这只是粗略兜底：同一数量级不代表语义相同
    #     （实测它会把 ExitRates(跳出率) 和 SpecialDay(节庆程度) 分到一组，
    #       虽然都是 0.1 量级，但语义无关）——所以还是手写 GROUPS 更靠谱
    _auto = {}
    _tmp_rows = full.loc[full['__is_train'] == 1, list(num_cols)]
    for _c in num_cols:
        _med = float(np.nanmedian(np.abs(_tmp_rows[_c].values.astype(float))))
        if _med <= 0:
            continue
        _mag = int(np.floor(np.log10(_med)))          # 0.02 -> -2；553 -> 2
        _auto.setdefault('mag%d' % _mag, []).append(_c)
    GROUPS = {k: v for k, v in _auto.items() if len(v) >= 2}
    if GROUPS:
        P('片段② 未填 GROUPS -> 按数量级粗略分组:')
        for _k, _v in sorted(GROUPS.items()):
            P('    %-7s %s' % (_k, _v))
        P('    ⚠ 同一量级 != 语义相同，建议手写 GROUPS（见片段② 注释里的例子）')
    else:
        GROUPS = {'all': list(num_cols)}
        P('   ⚠ 数量级都一样，退回"全部数值列一组"（这时 sum/mean 基本是噪声）')

for _gname, _cols in GROUPS.items():
    _cols = [c for c in _cols if c in full.columns]
    if len(_cols) < 2:
        continue
    _sub = full[_cols]
    _sfx = '_%s' % _gname
    full['rowsum' + _sfx] = _sub.sum(axis=1)
    full['rowmean' + _sfx] = _sub.mean(axis=1)
    full['rowstd' + _sfx] = _sub.std(axis=1)
    full['rowmax' + _sfx] = _sub.max(axis=1)
    full['rowmin' + _sfx] = _sub.min(axis=1)
    full['rowrange' + _sfx] = full['rowmax' + _sfx] - full['rowmin' + _sfx]
    full['rowcv' + _sfx] = full['rowstd' + _sfx] / (full['rowmean' + _sfx].abs() + 1e-6)
    _tot = full['rowsum' + _sfx].abs() + 1e-6
    for _c in _cols:                # 每个成员占本组总和的比例（结构特征，最有用）
        _name = '%s_pct%s' % (_c, _sfx)
        full[_name] = full[_c] / _tot
        new_feats.append(_name)
    new_feats += ['rowsum' + _sfx, 'rowmean' + _sfx, 'rowstd' + _sfx,
                  'rowmax' + _sfx, 'rowmin' + _sfx, 'rowrange' + _sfx, 'rowcv' + _sfx]
    P('片段② 组「%s」：7 个统计量 + %d 个占比' % (_gname, len(_cols)))


# ============================================================================
# 片段 ③ OOF 目标编码（中基数类别的提分利器）
# ============================================================================
# 【什么列用】类别取值 3~500 个：城市、职业、支付方式、套餐、渠道、邮编
# 【什么列别用】
#     · 取值只有 2 个  -> 用片段⑦⑨，目标编码没意义
#     · 取值上千个     -> 用片段④频次编码，目标编码会过拟合
# 【插到哪】模板「★ 特征片段粘贴处」
# 【铁律】必须折外(OOF)编码！直接 groupby 求均值 = 让模型偷看答案：
#         线下 F1 虚高、实际提交崩（最严重的错误之一）
# 【怎么选列】填进 TE_COLS；留空则自动挑"基数 3~500"的类别列
# ----------------------------------------------------------------------------
TE_COLS = [
    # 'PaymentMethod', 'InternetService', 'Contract', 'city',
]
if not TE_COLS:
    TE_COLS = [c for c in cat_cols if 3 <= full[c].nunique() <= 500]
P('\n片段③ 目标编码列:', TE_COLS)


def oof_target_encode(df, cat_col, target_col=TARGET, n_splits=5,
                      smoothing=20, seed=42):
    """折外目标编码，返回新列名（类别数不合适则返回 None）。"""
    tr_mask = df['__is_train'] == 1
    tr = df.loc[tr_mask, [cat_col, target_col]].copy()
    if not (3 <= tr[cat_col].nunique() <= 5000):
        return None
    y = tr[target_col].astype(int).values
    gm = y.mean()

    def _enc(keys, mean_map, cnt_map):
        m = pd.Series(keys).map(cnt_map).fillna(0).values
        mu = pd.Series(keys).map(mean_map).fillna(gm).values
        return (m * mu + smoothing * gm) / (m + smoothing)      # 贝叶斯平滑

    st = tr.groupby(cat_col)[target_col].agg(['mean', 'count'])
    out = np.full(len(df), np.nan)
    out[(~tr_mask).values] = _enc(df.loc[~tr_mask, cat_col].values, st['mean'], st['count'])
    tr_enc = np.zeros(len(tr))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for inn, outd in skf.split(tr, y):
        fs = tr.iloc[inn].groupby(cat_col)[target_col].agg(['mean', 'count'])
        tr_enc[outd] = _enc(tr.iloc[outd][cat_col].values, fs['mean'], fs['count'])
    out[np.where(tr_mask.values)[0]] = tr_enc
    col = cat_col + '_te'
    df[col] = out
    return col


for _c in TE_COLS:
    _col = oof_target_encode(full, _c)
    if _col:
        keep_feats.append(_col)
        P('  目标编码: %s -> %s' % (_c, _col))


# ============================================================================
# 片段 ④ 频次编码（高基数类别 / ID 类字段）
# ============================================================================
# 【什么列用】取值很多：CustomerID、邮编、商品码、设备号、IP、商家ID
# 【什么列别用】取值很少（2~3 类）-> 频次就那几个值，没信息
# 【插到哪】模板「★ 特征片段粘贴处」
# 【为什么安全】只用特征、不看标签，可以 train+test 一起统计，不会泄漏
# ----------------------------------------------------------------------------
FREQ_MIN_UNIQ = 10         # 取值数超过它才做频次编码
_hi = [c for c in cat_cols if full[c].nunique() > FREQ_MIN_UNIQ]
for _c in _hi:
    _vc = full[_c].value_counts()
    full[_c + '_freq'] = full[_c].map(_vc).astype(float)
    new_feats.append(_c + '_freq')
    full[_c + '_freqpct'] = (full[_c].map(_vc) / len(full)).astype(float)
    new_feats.append(_c + '_freqpct')
P('片段④ 完成：高基数列 %d 个（> %d 个取值）' % (len(_hi), FREQ_MIN_UNIQ))


# ============================================================================
# 片段 ⑤ 有序类别映射（别让 LabelEncoder 按字母序乱编）
# ============================================================================
# 【什么列用】有业务顺序的文本列：
#     · Contract：Month-to-month < One year < Two year
#     · Month：Jan < Feb < ... < Dec（月份本身含时间序）
#     · 学历：小学 < 初中 < 高中 < 本科 < 硕士 < 博士
#     · 满意度 / 等级：low < mid < high；A > B > C > D
# 【为什么必须手写】LabelEncoder 按字母序编码，会把 Dec 编得比 Jan 小、
#     把 Two year 编得比 One year 小 —— 顺序含义直接错掉
# 【插到哪】模板「★ 特征片段粘贴处」（要排在类别编码之前）
# ----------------------------------------------------------------------------
ORDINAL_MAPS = {
    # 列名: {取值: 顺序号}
    # 'Contract':    {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    # 'Month':       {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    #                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12},
    # 'Education':   {'Primary': 0, 'Secondary': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4},
    # 'Level':       {'low': 0, 'mid': 1, 'high': 2},
    # 'CreditGrade': {'D': 0, 'C': 1, 'B': 2, 'A': 3},
}

# 月份列自动识别（取值像月份就自动做，不用手写）
_MONTH = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'june': 6,
          'jul': 7, 'july': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10,
          'nov': 11, 'dec': 12, 'december': 12}
for _c in cat_cols:
    _vals = set(full[_c].astype(str).str.lower().unique())
    if _vals and _vals <= set(_MONTH):
        full[_c + '_mon'] = full[_c].str.lower().map(_MONTH).astype(float)
        new_feats.append(_c + '_mon')
        P('  月份自动识别: %s -> %s_mon' % (_c, _c))

for _c, _mp in ORDINAL_MAPS.items():
    if _c in full.columns:
        full[_c + '_ord'] = full[_c].map(_mp).astype(float)
        new_feats.append(_c + '_ord')
        P('  有序映射: %s -> %s_ord' % (_c, _c))

# 检查：映射表漏了取值的话，会变成 NaN，一定要看一眼
for _c, _mp in ORDINAL_MAPS.items():
    if _c in full.columns:
        _missed = set(full.loc[full[_c + '_ord'].isna(), _c].unique()) - {'__MISSING__'}
        if _missed:
            P('  ⚠ %s 有取值没写进映射表（已变缺失）: %s' % (_c, list(_missed)[:5]))


# ============================================================================
# 片段 ⑥ 分箱（连续变量的非线性）
# ============================================================================
# 【什么列用】
#     · 和目标非单调的连续变量（年龄、收入：中间段风险最高）
#     · 天然该分档的（金额档、停留时长档）
#     · 极端值多、想弱化其影响的列
# 【什么列别用】和目标已近似单调的列（分箱反而丢信息，让树自己切更好）
# 【插到哪】模板「★ 特征片段粘贴处」
# 【档数】5~10 档足够；qcut=等频分箱，cut=等宽分箱
# ----------------------------------------------------------------------------
BIN_COLS = [
    # ('age', 5), ('MonthlyCharges', 10), ('TotalCharges', 10),
]
for _c, _q in BIN_COLS:
    if _c in full.columns:
        full['%s_bin%d' % (_c, _q)] = pd.qcut(full[_c], _q, labels=False,
                                              duplicates='drop').astype(float)
        new_feats.append('%s_bin%d' % (_c, _q))
        P('  分箱: %s -> %s_bin%d' % (_c, _c, _q))

# 【自动版】直接按"目标率跨箱是否稳定变化"判断（比互信息更稳、更快、更好解释）
#   为什么不用互信息：实测过，k-NN 互信息对非单调关系本来就能捕捉（如 U 型的年龄
#   原始互信息已经 0.0789），分箱反而丢信息；而且列的信息量接近 0 时
#   "提升倍数"会数学上爆掉（0.003/0 = 无穷大）—— 判据不可靠。
#   现在的判据：把列等频分 10 箱 -> 算每箱的正类率 -> 看这些比率的波动大小。
#   波动大 = 目标率随该列明显变化 = 值得分箱（树模型也更容易切）。
#   ⚠ 说句实话：对 GBDT 类模型，分箱通常帮助有限（树自己会切分），
#     所以这个片段是"锦上添花"，别指望它大幅提分。
if not BIN_COLS:
    try:
        _tr2 = full['__is_train'] == 1
        _ytr = full.loc[_tr2, TARGET].astype(int).values
        _p0 = float(_ytr.mean())
        for _c in [c for c in num_cols if full[c].nunique() > 20][:12]:
            _b = pd.qcut(full.loc[_tr2, _c], 10, labels=False, duplicates='drop')
            if _b.nunique() < 4:
                continue
            _rate = pd.Series(_ytr).groupby(_b.values).mean()
            _sz = pd.Series(_ytr).groupby(_b.values).size()
            if _sz.min() < 30:                      # 每箱样本太少，比率不可信
                continue
            _sd = float(_rate.std())
            if _sd > 0.03 and _sd / max(_p0, 1e-9) > 0.15:   # 目标率波动明显
                full['%s_bin10' % _c] = pd.qcut(full[_c], 10, labels=False,
                                                duplicates='drop').astype(float)
                new_feats.append('%s_bin10' % _c)
                P('  自动分箱: %-22s 跨箱正类率标准差=%.4f（整体 %.4f）'
                  % (_c, _sd, _p0))
        if not any(str(f).endswith('_bin10') for f in new_feats):
            P('  自动分箱: 没有列的"跨箱目标率"波动够大 -> 跳过（对树模型很常见，正常）')
    except Exception as _e:
        P('  自动分箱跳过:', type(_e).__name__, str(_e)[:60])


# ============================================================================
# 片段 ⑦ 类别交叉（组合特征）
# ============================================================================
# 【什么列用】两个低基数列的"组合"有业务含义：
#     · 合约类型 × 支付方式   -> 不同组合流失倾向差别很大
#     · 城市 × 套餐           -> 区域偏好
#     · 渠道 × 设备           -> 转化差异
# 【什么列别用】两个高基数列交叉（组合数爆炸 = 给每行一个独立编码）
# 【插到哪】模板「★ 特征片段粘贴处」
# 【注意】只做基数<20 的列，最多几组；交叉后可以再接目标编码
# ----------------------------------------------------------------------------
CROSS_PAIRS = [
    # ('Contract', 'PaymentMethod'),
    # ('InternetService', 'OnlineSecurity'),
]
if not CROSS_PAIRS:
    # 自己 import，不依赖片段①（每个片段都必须能单独粘贴使用）
    from itertools import combinations as _comb7
    _low = [c for c in cat_cols if full[c].nunique() <= 12][:4]
    CROSS_PAIRS = list(_comb7(_low, 2))[:3]
    P('\n片段⑦ 自动挑的交叉组合:', CROSS_PAIRS)

for _a, _b in CROSS_PAIRS:
    if _a in full.columns and _b in full.columns \
            and full[_a].nunique() < 20 and full[_b].nunique() < 20:
        _key = '%s_x_%s' % (_a, _b)
        full[_key] = full[_a].astype(str) + '|' + full[_b].astype(str)
        full[_key + '_le'] = LabelEncoder().fit_transform(full[_key]).astype(float)
        new_feats.append(_key + '_le')
        P('  交叉: %s（基数 %d）' % (_key, full[_key].nunique()))


# ============================================================================
# 片段 ⑧ 日期 / 时间特征
# ============================================================================
# 【什么列用】日期或时间戳：注册日期、下单时间、账单月、最后登录时间
# 【什么列别用】"月份"已被编码成 1~12 的列（那用片段⑤）
# 【插到哪】模板「★ 特征片段粘贴处」（越早越好，后面还能基于它造特征）
# 【重点】相对时间（距今天数）通常比绝对日期（2026-09-17）有用，
#     因为测试集里绝对年份是模型没见过的
# ----------------------------------------------------------------------------
DATE_COLS = [
    # 'order_date', 'signup_time', 'bill_date', 'last_login',
]
if not DATE_COLS:                      # 自动找：列名像日期 + 能解析成功
    for _c in list(full.columns):
        if _c in (TARGET, ID_COL, '__is_train') or full[_c].dtype != object:
            continue
        if not any(k in str(_c).lower() for k in ['date', 'time', '日期', '时间']):
            continue
        if pd.to_datetime(full[_c], errors='coerce').notna().mean() > 0.8:
            DATE_COLS.append(_c)
    if DATE_COLS:
        P('\n片段⑧ 自动识别到的日期列:', DATE_COLS)

for _c in DATE_COLS:
    _d = pd.to_datetime(full[_c], errors='coerce')
    if _d.notna().mean() < 0.5:
        P('  跳过 %s（解析成功率太低）' % _c)
        continue
    full[_c + '_month'] = _d.dt.month.astype(float)
    full[_c + '_day'] = _d.dt.day.astype(float)
    full[_c + '_dow'] = _d.dt.dayofweek.astype(float)                   # 0=周一
    full[_c + '_is_weekend'] = (_d.dt.dayofweek >= 5).astype(float)
    full[_c + '_quarter'] = _d.dt.quarter.astype(float)
    full[_c + '_elapsed'] = (_d - _d.min()).dt.days.astype(float)       # 距最早日期
    full[_c + '_since_last'] = (_d.max() - _d).dt.days.astype(float)    # 距最晚日期
    new_feats += [_c + '_month', _c + '_day', _c + '_dow', _c + '_is_weekend',
                  _c + '_quarter', _c + '_elapsed', _c + '_since_last']
    P('  日期特征: %s（解析成功 %.1f%%）' % (_c, 100 * _d.notna().mean()))


# ============================================================================
# 片段 ⑨ log1p / 幂变换（长尾分布）
# ============================================================================
# 【什么列用】右偏严重的长尾列：金额、停留时长、点击数、浏览量
#     判断标准：skew() > 2（下面代码自动算）
# 【什么列别用】已经是比率(0~1)、或计数分布均匀的列
# 【插到哪】模板「★ 特征片段粘贴处」
# 【说明】树模型对单调变换不敏感（提升有限），但做比率特征和线性模型时有用，
#     成本极低，顺手加上没坏处
# ----------------------------------------------------------------------------
for _c in list(num_cols):
    _u = full.loc[full['__is_train'] == 1, _c].dropna()
    if len(_u) < 100:
        continue
    if _u.min() >= 0 and _u.skew() > 2:
        full[_c + '_log'] = np.log1p(full[_c])
        new_feats.append(_c + '_log')
        P('  log1p: %s（skew=%.1f）' % (_c, _u.skew()))


# ============================================================================
# 片段 ⑩ 异常打分（当"异常本身"是信号时）
# ============================================================================
# 【什么列用】不确定就全用数值列；典型场景：欺诈检测、故障预警、违约预测
#     —— "不正常"往往就是信号
# 【插到哪】模板「★ 特征片段粘贴处」；也可以用模板内置开关 USE_ISOLATION=True
# 【代价】2.4 万行约 10 秒；有没有用看模板「加分实验 C」的对比
# ----------------------------------------------------------------------------
from sklearn.ensemble import IsolationForest
_iso_cols = [c for c in num_cols if full[c].nunique() > 5]
if len(_iso_cols) >= 3:
    _iso = IsolationForest(n_estimators=200, contamination='auto',
                           random_state=42, n_jobs=4)
    _iso.fit(full.loc[full['__is_train'] == 1, _iso_cols])
    full['iso_score'] = -_iso.score_samples(full[_iso_cols])            # 越大越异常
    full['iso_flag'] = (_iso.predict(full[_iso_cols]) == -1).astype(float)
    new_feats += ['iso_score', 'iso_flag']
    P('片段⑩ 完成：异常比例 %.4f' % full['iso_flag'].mean())


# ============================================================================
# 片段 ⑪ 文本列特征（结构化题里混了短文本时）
# ============================================================================
# 【什么列用】自由文本但不算长：备注、商品标题、姓名、地址
# 【什么列别用】规则类别（Yes/No）-> 那是片段⑦⑨ 的活
# 【插到哪】模板「★ 特征片段粘贴处」
# 【思路】不训练文本模型（考场没时间），只抽"长度/词数/是否含数字"这类便宜特征；
#     反复出现的占位词（'n/a'、'unknown'）加个 0/1 关键词特征也有用
# ----------------------------------------------------------------------------
TEXT_COLS = [
    # 'remark', 'product_name', 'description',
]
if not TEXT_COLS:
    # 自动找"疑似文本"的列，排除日期列（日期该用片段⑧，别当文本处理）
    for _c in cat_cols:
        if full[_c].nunique() <= 50:
            continue
        if full[_c].astype(str).str.len().mean() <= 8:
            continue
        _n = str(_c).lower()
        if any(k in _n for k in ['date', 'time', '日期', '时间']):
            continue
        if pd.to_datetime(full[_c], errors='coerce').notna().mean() > 0.8:
            continue                                   # 能解析成日期的，交给片段⑧
        TEXT_COLS.append(_c)
    TEXT_COLS = TEXT_COLS[:5]
    if TEXT_COLS:
        P('\n片段⑪ 自动识别到的疑似文本列:', TEXT_COLS)

KEYWORDS = ['n/a', 'unknown', 'none', 'null', 'error', 'test', 'vip', 'free']
for _c in TEXT_COLS:
    if _c not in full.columns:
        continue
    _s = full[_c].astype(str)
    full[_c + '_len'] = _s.str.len().astype(float)
    full[_c + '_nwords'] = _s.str.count(r'\s+').astype(float) + 1
    full[_c + '_ndigit'] = _s.str.count(r'\d').astype(float)
    full[_c + '_hasdigit'] = _s.str.contains(r'\d', regex=True).astype(float)
    full[_c + '_nuniqchar'] = _s.map(lambda x: len(set(x))).astype(float)
    full[_c + '_nkw'] = _s.str.lower().apply(
        lambda x: sum(k in x for k in KEYWORDS)).astype(float)
    new_feats += [_c + '_len', _c + '_nwords', _c + '_ndigit',
                  _c + '_hasdigit', _c + '_nuniqchar', _c + '_nkw']
    P('  文本特征: %s' % _c)


# ============================================================================
# 附一：粘贴完之后的检查清单（4 条，别跳）
# ============================================================================
# 1) 回模板看「★ 特征 N 个」——N 应明显大于原始列数；
#    N 没变说明片段没生效（常见原因：列名写错、变量名打错）
# 2) 模板会自动做三件保护：去重、删常量列、训练/测试列数一致性断言
# 3) 打开模板「加分实验 C」看对比：仅原始列 vs 加衍生特征，确认真的提分了
# 4) 特征太多（>300）会拖慢训练：把片段① 的 PAIR_TOP_N 调小，或少贴几个片段


# ============================================================================
# 附二：按真题类型的推荐组合（照抄即可）
# ============================================================================
# 【电商行为数据（2026.06 那种）】
#   ① Product_Duration/Admin_Duration、TotalCharges/tenure 这类比率
#   ② 页面数一组、时长一组 -> 再取占比
#   ⑤ Month 月份顺序
#   ⑥ SpecialDay / PageValues 分箱
#   => 主特征是"停留时长 × 页面数 × 跳出率"的组合
#
# 【电信流失数据（2026.03 那种）】
#   ① TotalCharges/MonthlyCharges（=已用月数）、MonthlyCharges/tenure（=月均）
#   ② 6 个附加服务（OnlineSecurity/OnlineBackup/...）一组求和
#   ⑤ Contract 的 月付<1年<2年 顺序
#   ⑥ tenure 分箱（0-12/12-24/24-48/48-72/72+）
#   ⑨ TotalCharges 做 log1p
#   => 主特征是"合约类型 × 附加服务数量 × 月费水平"
#
# 【有日期/时间戳的数据】
#   ⑧ 优先做，再基于它做①（如 订单数/距注册天数）
#
# 【类别特别多 + 疑似 ID】
#   ④ 频次编码 + ③ 目标编码；不要用 LabelEncoder 硬编
# ============================================================================


# ============================================================================
# 附三：命名约定与"别做的事"
# ============================================================================
# 【命名】衍生列带上后缀，一眼能看出是什么：
#   __div__ / __sub__（比率差值） _pct（占比） _freq（频次） _te（目标编码）
#   _bin10（分箱） _log（对数） _ord（顺序） _mon（月份）
#   —— 模板「加分实验 C」靠命名里是否带 "__" 区分原始列与衍生列
#
# 【往哪塞】
#   new_feats  -> 数值类新特征（模板会收，且参与后续筛选语境）
#   keep_feats -> 类别编码类（目标编码、有序编码、交叉编码）
#   两者模板都会收进模型，区别只是语义上的
#
# 【别做】
#   · 别把 ID / 流水号当特征（无预测意义，还容易过拟合）
#   · 别在拆分 train/test 之后分别造特征（列数会对不上，PPT 明确要求一致）
#   · 别对目标编码偷懒用全量均值（必须 OOF，见片段③）
# ============================================================================
