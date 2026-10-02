# -*- coding: utf-8 -*-
# 由 16_自动检测该做什么特征工程.py 生成
# 用法：整段内容复制到 12_模板_202609结构化实战模板.py 的「★ 特征片段粘贴处」

# ============================================================================
# 自动生成的特征工程代码 —— 整段复制到 12 模板的「★ 特征片段粘贴处」
#   生成时间: 2026-09-18 19:39:34
#   数据: ecommerce_train.csv
#   目标列: Revenue | ID列: ID | 训练行 30000 | 字段 20
# ============================================================================

# ---- 片段① 比率/差值（只做实测互信息最高的 8 对，避免噪声）----
PAIRS = [
    ('ProductRelated', 'Product_Duration'),   # 互信息 0.01235
    ('Administrative', 'Product_Duration'),   # 互信息 0.00829
    ('Informational', 'Product_Duration'),   # 互信息 0.00472
    ('Admin_Duration', 'Product_Duration'),   # 互信息 0.00449
    ('Info_Duration', 'ProductRelated'),   # 互信息 0.00404
    ('Product_Duration', 'PageValues'),   # 互信息 0.00334
    ('ProductRelated', 'PageValues'),   # 互信息 0.00332
    ('Info_Duration', 'PageValues'),   # 互信息 0.00331
]
for _a, _b in PAIRS:
    if _a not in full.columns or _b not in full.columns:
        continue
    _v = full[_a].astype(float) / (full[_b].astype(float) + 1e-6)
    _v = np.where(np.isfinite(_v), _v, np.nan)
    _col = '%s__div__%s' % (_a, _b)
    full[_col] = np.where(np.isnan(_v), np.nanmedian(_v), _v)
    full[_col] = full[_col].clip(*np.nanpercentile(_v, [0.5, 99.5]))
    new_feats.append(_col)
    _col2 = '%s__sub__%s' % (_a, _b)
    full[_col2] = full[_a].astype(float) - full[_b].astype(float)
    new_feats.append(_col2)
print('片段① 完成：%d 对 -> %d 列' % (len(PAIRS), len(PAIRS) * 2))

# ---- 片段② 行统计（按同类列分组，自动找到 1 组）----
GROUPS = {
    'duration': ['Admin_Duration', 'Info_Duration', 'Product_Duration'],
}
for _gname, _cols in GROUPS.items():
    _cols = [c for c in _cols if c in full.columns]
    if len(_cols) < 2:
        continue
    _sfx = '_%s' % _gname
    _sub = full[_cols]
    full['rowsum' + _sfx] = _sub.sum(axis=1)
    full['rowmean' + _sfx] = _sub.mean(axis=1)
    full['rowstd' + _sfx] = _sub.std(axis=1)
    full['rowmax' + _sfx] = _sub.max(axis=1)
    full['rowmin' + _sfx] = _sub.min(axis=1)
    full['rowrange' + _sfx] = full['rowmax' + _sfx] - full['rowmin' + _sfx]
    full['rowcv' + _sfx] = full['rowstd' + _sfx] / (full['rowmean' + _sfx].abs() + 1e-6)
    _tot = full['rowsum' + _sfx].abs() + 1e-6
    for _c in _cols:
        _n = '%s_pct%s' % (_c, _sfx)
        full[_n] = full[_c] / _tot
        new_feats.append(_n)
    new_feats += ['rowsum' + _sfx, 'rowmean' + _sfx, 'rowstd' + _sfx,
                  'rowmax' + _sfx, 'rowmin' + _sfx, 'rowrange' + _sfx, 'rowcv' + _sfx]
    print('片段② 组「%s」：7 个统计量 + %d 个占比' % (_gname, len(_cols)))

# ---- 片段⑤ 有序映射 ----
# 月份列（模板已能自动识别，这里保留是为了更清楚）: ['Month']
# 需要你手写顺序表的列（脚本猜不出业务顺序，请按业务实际填写）:
# ORDINAL_MAPS['Month'] = {...}   # 例: {'low':0, 'mid':1, 'high':2}
_MONTH_MAP = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'june':6,'jul':7,
              'july':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12}
for _c in cat_cols:
    _vals = set(full[_c].astype(str).str.strip().str.lower().unique())
    if _vals and _vals <= set(_MONTH_MAP):
        full[_c + '_mon'] = full[_c].astype(str).str.lower().map(_MONTH_MAP).astype(float)
        new_feats.append(_c + '_mon')
        print('片段⑤ 月份顺序编码: %s' % _c)

# ---- 片段③ OOF 目标编码（中/高基数类别）----
TE_COLS = ['VisitorType', 'User_Horoscope']
def _oof_te(df, col, target_col=TARGET, n_splits=5, smoothing=20, seed=SEED):
    _trm = df['__is_train'] == 1
    _tr = df.loc[_trm, [col, target_col]].copy()
    if _tr[col].nunique() < 2:
        return None
    _y = _tr[target_col].astype(int).values
    _gm = _y.mean()
    def _enc(keys, mm, cm):
        m = pd.Series(keys).map(cm).fillna(0).values
        mu = pd.Series(keys).map(mm).fillna(_gm).values
        return (m * mu + smoothing * _gm) / (m + smoothing)
    _st = _tr.groupby(col)[target_col].agg(['mean', 'count'])
    _out = np.full(len(df), np.nan)
    _out[(~_trm).values] = _enc(df.loc[~_trm, col].values, _st['mean'], _st['count'])
    _tre = np.zeros(len(_tr))
    _skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for _in, _od in _skf.split(_tr, _y):
        _fs = _tr.iloc[_in].groupby(col)[target_col].agg(['mean', 'count'])
        _tre[_od] = _enc(_tr.iloc[_od][col].values, _fs['mean'], _fs['count'])
    _out[np.where(_trm.values)[0]] = _tre
    df[col + '_te'] = _out
    return col + '_te'

for _c in TE_COLS:
    _n = _oof_te(full, _c)
    if _n:
        new_feats.append(_n)
        print('片段③ 目标编码: %s -> %s' % (_c, _n))

# ---- 片段④ 频次编码（只给真正的高基数"类别/编码"，不给连续量用）----
# 没有"高基数的类别列"（连续量的唯一值多但不算高基数类别）
# 如果模板打印 "⚠ XXX 基数 N 过大，跳过编码"，再把那列名填进 FREQ_COLS 手动开。
# FREQ_COLS = []

# ---- 片段⑥ 分箱（只对偏度大的连续列）----
# 没有偏度 > 2 的连续列，跳过片段⑥

# ---- 片段⑧ 日期特征 ----
# 没有日期列，跳过片段⑧

# ---- 片段⑪ 文本特征 ----
# 没有短文本列，跳过片段⑪

# ---- 片段⑨ log 变换（长尾量列）----
LOG_COLS = ['Admin_Duration']
for _c in LOG_COLS:
    if _c in full.columns:
        full[_c + '_log'] = np.log1p(full[_c].clip(lower=0))
        new_feats.append(_c + '_log')
        print('片段⑨ log1p: %s' % _c)

# ---- 片段⑩ 异常打分：默认不贴 ----
# 实测在弱信号数据上会掉分（iso_score 单独 AUC 只有 0.54）。
# 只有题目背景是"欺诈/故障/违约"这类"异常即目标"的场景才建议开。
# 想试就把模板 cell 0 的 USE_ISOLATION 改成 True。

# ============================================================================
# ⚠ 粘完之后务必做这一步（30 秒，决定要不要保留这些特征）
# ============================================================================
#   打开 12 模板 cell 23，把 COMPARE_FE 改成 True 再跑一次，看这两行：
#       仅原始字段      : OOF F1 = xxxx  (特征数 N1)
#       原始+衍生特征   : OOF F1 = yyyy  (特征数 N2)
#   ★ yyyy 明显大于 xxxx 才保留本文件；否则把这段删掉，用原始字段交卷。
#   为什么必须验证：实测在信号弱的数据（AUC 只有 0.6 左右）上，
#   加特征可能反而降分（我在这份电商数据上实测过：基线 0.2716 -> 加特征 0.2619）。
#   数据里若存在"必须自己构造才能表达"的关系（如 总消费/月数 = 已用月数），
#   加特征才会明显提分。
# ============================================================================

print('★ 自动生成的特征工程完成，新增列数:', len(new_feats))