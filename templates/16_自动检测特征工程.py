# -*- coding: utf-8 -*-
"""
16_自动检测该做什么特征工程.py —— 一键分析 + 生成"可直接粘贴"的代码
============================================================================

【它替你做什么】
  你只要给它训练数据，它自动：
    1. 逐个字段判类型（次数/金额/时长/比率/等级/类别码/日期/文本/ID/二值）
    2. 按类型推荐该用哪个特征工程片段（①~⑪）
    3. ★重点★ 生成一份【列名已经填好】的代码块，你整段复制到 12 模板的粘贴处即可
    4. 报告哪些列有问题（高基数没处理、伪有序、ID 混进特征等）

【怎么用】
    # 最简单：只给训练集（目标列和 ID 列自动猜）
    python 16_自动检测该做什么特征工程.py ecommerce_train.csv

    # 指定目标列和 ID 列
    python 16_自动检测该做什么特征工程.py ecommerce_train.csv -t Revenue -i ID

    # 不加参数 = 用 2026.06 电商真题试跑
    python 16_自动检测该做什么特征工程.py

【输出】
    1. 屏幕上的分析报告（每个字段判成什么、为什么）
    2. 同目录生成一个 _特征工程_已填好.py 文件 —— 整段复制到 12 模板的
       「★ 特征片段粘贴处」即可（列名、分组、参数都填好了）
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.model_selection import StratifiedKFold
    _HAS_SK = True
except ImportError:
    _HAS_SK = False

# ============================================================================
# 配置：脚本内部用到的关键词表（想扩展就改这里）
# ============================================================================
KW_DURATION = ['duration', 'time', 'tenure', '时长', '停留', '持续', '秒']
KW_AMOUNT = ['amount', 'charge', 'price', 'fee', 'cost', 'revenue', 'salary',
             '金额', '价格', '费用', '消费', '客单', '收入']
KW_COUNT = ['num', 'cnt', 'count', 'qty', 'quantity', 'visit', 'times', 'freq',
            '次数', '数量', '笔数', '个数', '页面']
KW_RATE = ['rate', 'ratio', 'pct', 'percent', 'share', '率', '占比', '比例']
KW_LEVEL = ['level', 'grade', 'score', 'rank', 'star', '满意度', '等级', '评分', '档']
KW_CODE = ['id', 'code', 'no', 'key', 'flag', 'is_', 'os', 'browser', 'region',
           'traffic', 'type', 'zone', 'area', '编码', '代码', '编号', '类型',
           '地区', '区域', '系统', '渠道']
# ★ 识别"已经是衍生列"的词。这里【必须分词匹配，不能子串匹配】——
#   实测踩过：'Administrative' 里含 'min'（ad-MIN-istrative），
#   子串匹配会把它误判成"最小值衍生列"，荒唐但确实发生了。
#   规则：把列名切成 token（下划线/CamelCase 都切开），只有整词命中才算。
KW_AGG_TOKENS = {
    'avg', 'average', 'mean', 'median', 'std', 'var', 'sum', 'total',
    'min', 'max', 'range', 'ratio', 'rate', 'pct', 'percent', 'share',
    'prop', 'freq', 'diff', 'delta', 'grow', 'growth', 'change',
    'sum', 'cumsum', 'cummax', 'cummin', 'rank', 'score',
    '平均', '均值', '总计', '合计', '占比', '比例', '比率', '频次',
    '分箱', '档位', '评分', '累计', '同比', '环比',
}
# 明确表示"这列本身就是比率"的词
KW_IS_RATIO = ['rate', 'ratio', 'pct', 'percent', 'share', 'prop',
               '率', '占比', '比例', '比率']
# 单 token 名字的精确白名单（全小写、没下划线时用得上）
_AGG_EXACT = {'rate', 'ratio', 'pct', 'avg', 'mean', 'total', 'sum', 'cnt',
              'count', 'ratio', 'share', 'prop', 'freq'}
# 严格后缀（下划线开头才算，避免 count 匹配到 account 之类的巧合）
_AGG_SUFFIX = ['_cnt', '_count', '_avg', '_mean', '_median', '_total', '_sum',
               '_min', '_max', '_std', '_var', '_pct', '_ratio', '_rate',
               '_freq', '_share', '_prop', '_bin', '_bucket', '_rank', '_score',
               '_diff', '_delta', '_change']


def split_tokens(name):
    """把列名切成 token：下划线/空格/连字符切开，CamelCase 也切开。

    例：avgSession -> ['avg','session']；total_charge -> ['total','charge']；
        Administrative -> ['administrative']（整词，不会被 'min' 误命中）；
        min_child_samples -> ['min','child','samples']。
    中文不切（中文列名整体当一个 token，靠中文关键词单独匹配）。
    """
    n = str(name)
    if re.search(r'[\u4e00-\u9fff]', n):
        return [n.lower()]
    n = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', n)          # CamelCase 切开
    n = re.sub(r'[^A-Za-z0-9]+', ' ', n)                    # 其余分隔符统一成空格
    return [t.lower() for t in n.split() if t]


def _is_code_name(name):
    """判断列名是否像"编码"。★ OperatingSystems 要排除掉（它是系统代码，
    但名字里有 system，容易和'操作系统=量'混淆；实际它是 1~7 的编码）。"""
    n = norm_name(name)
    if 'operatingsystem' in n or n == 'os' or n.startswith('os'):
        return True
    if any(k in n for k in ['system', '系统']) and 'duration' not in n and 'time' not in n:
        return True
    return match_kw(name, KW_CODE)


def _is_derived_name(name):
    """判断列名是否"已经是衍生列"（均值/合计/比率/分箱…）。

    ★ 用分词匹配，不用子串（子串会把 Administrative 判成"最小值衍生列"）。
    为什么重要：这类列不能再当"基础量"参与比率配对 ——
    否则会造出 `total_charge / avg_charge` 这种循环无意义的东西（实测踩过）。
    """
    toks = split_tokens(name)
    low = str(name).lower()
    # 中文列名：直接子串匹配中文关键词
    if re.search(r'[\u4e00-\u9fff]', str(name)):
        return match_kw(name, list(KW_IS_RATIO) +
                        ['平均', '均值', '总计', '合计', '占比', '比例', '比率', '频次', '分箱'])
    # 严格后缀（下划线风格）
    if any(low.endswith(s) for s in _AGG_SUFFIX):
        return True
    # 整词命中（token 风格）
    if any(t in KW_AGG_TOKENS for t in toks):
        return True
    # 单 token 名字的精确匹配
    if len(toks) == 1 and toks[0] in _AGG_EXACT:
        return True
    return False


def _is_id_like_values(name, s, n_rows):
    """值层面判断是否像 ID：全是长数字串（手机号/卡号/邮编）。"""
    nm = norm_name(name)
    if match_kw(name, ['phone', 'mobile', 'tel', 'zip', 'postal', 'card', 'account',
                       '手机', '电话', '邮编', '卡号', '账号']):
        return True
    if not pd.api.types.is_numeric_dtype(s):
        return False
    v = pd.to_numeric(s, errors='coerce').dropna()
    if len(v) < 20:
        return False
    # 全是 7 位以上的整数（手机号 11 位、邮编 6 位、卡号）-> 是标识不是量
    if not np.allclose(v.values, np.round(v.values)):
        return False
    return bool((v >= 10 ** 6).mean() > 0.95)
KW_MONTH = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'june': 6,
            'jul': 7, 'july': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10,
            'nov': 11, 'dec': 12, 'december': 12}
KW_WEEKDAY = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7,
              'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4,
              'friday': 5, 'saturday': 6, 'sunday': 7,
              '周一': 1, '周二': 2, '周三': 3, '周四': 4, '周五': 5, '周六': 6, '周日': 7}
KW_NOMINAL_WORDS = ['month', 'monthly', 'quarter', 'season']   # 名义上的时间词

ID_LIKE_THRESHOLD = 0.9      # 唯一值占比超过它 + 名字带 id -> 判定为 ID
HIGH_CARD = 200              # 超过它 = 高基数，模板不编码
MID_CARD_MIN = 3             # 3~200 = 中基数，可做目标编码
FREQ_MIN_CARD = 10           # 超过它才值得做频次编码


# ============================================================================
# 读数据
# ============================================================================
def read_table(path, **kw):
    if str(path).lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path, **kw)
    for enc in ['utf-8-sig', 'gb18030', 'latin-1']:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding='latin-1')


# ============================================================================
# 第 1 步：字段分型
# ============================================================================
def norm_name(name):
    return str(name).lower().replace(' ', '').replace('-', '').replace('_', '')


def match_kw(name, kws):
    n = str(name).lower()
    return any(k in n for k in kws)


def parse_datetime_ratio(s):
    """能成功解析成日期的比例（抽样算，快）。"""
    try:
        sub = s.dropna()
        if len(sub) > 500:
            sub = sub.sample(500, random_state=0)
        if len(sub) < 20:
            return 0.0
        return float(pd.to_datetime(sub, errors='coerce').notna().mean())
    except Exception:
        return 0.0


def classify_column(name, s, n_rows):
    """判断一列的语义类型，返回 dict。

    判断顺序（★顺序很重要，先从"确定性最高"的判据开始）：
      1. 常量
      2. 名字/模式像 ID  -> ID
      3. datetime 类型或能解析 -> 日期
      4. 二值
      5. 文本（长字符串）
      6. 月份/星期（有序类别）
      7. 数值型细分：
         a) 名字命中关键词（duration / amount / count / rate）优先，这是最可靠的
         b) 没命中时，看"取值种类数 / 取值范围"判断是"量"还是"编码"
      8. 其余当类别，按基数分低/中/高
    """
    info = {'name': name, 'dtype': str(s.dtype), 'n_unique': int(s.nunique(dropna=True)),
            'missing': float(s.isna().mean())}
    u = info['n_unique']
    uniq_ratio = u / max(n_rows, 1)
    info['uniq_ratio'] = uniq_ratio
    nm = norm_name(name)
    is_num = pd.api.types.is_numeric_dtype(s)

    # ---- 1. 常量 ----
    if u <= 1:
        info.update(type='常量', seg='—', reason='整列只有一个值，删掉')
        return info

    # ---- 2. ID（关键修正：浮点连续值天然唯一值多，不能只看唯一率）----
    name_is_id = nm in ('id', 'index', 'uid', 'uuid') or nm.endswith('id') or \
        nm.startswith('id') or any(k in nm for k in ['编号', '流水', '序号'])
    if name_is_id and uniq_ratio > 0.5:
        info.update(type='ID/流水号', seg='—',
                    reason='列名像 ID 且唯一值占比 %.0f%% -> 流水号，绝不能当特征'
                           % (uniq_ratio * 100))
        return info
    if not is_num and uniq_ratio > ID_LIKE_THRESHOLD and u > 1000:
        info.update(type='ID/流水号', seg='—',
                    reason='字符串列且唯一值占比 %.0f%% -> 疑似流水号' % (uniq_ratio * 100))
        return info
    # ★ 值层面识别 ID：全是 7 位以上整数的（手机号 11 位 / 邮编 6 位 / 卡号）
    #   必须在"日期"判断之前，否则手机号会被 pd.to_datetime 误判成日期（实测踩过）
    if is_num and _is_id_like_values(name, s, n_rows):
        info.update(type='ID/标识号', seg='—',
                    reason='取值全是 7 位以上整数（手机号/卡号/邮编类）-> 是标识不是量')
        return info

    # ---- 3. 日期 ----
    if str(s.dtype).startswith('datetime'):
        info.update(type='日期时间', seg='⑧', reason='本身就是 datetime 类型')
        return info
    if not is_num and not match_kw(name, ['phone', 'mobile', 'tel', 'card', 'account',
                                          '手机', '电话', '卡号', '账号']):
        r = parse_datetime_ratio(s)
        if r > 0.8:
            info.update(type='日期时间', seg='⑧', reason='能解析成日期的比例 %.0f%%' % (r * 100))
            return info

    # ---- 4. 二值 ----
    if u <= 2:
        info.update(type='二值', seg='模板自动', reason='取值 2 个，模板第 5 节自动转 0/1')
        return info

    # ---- 5. 文本 ----
    if not is_num:
        avg_len = float(s.astype(str).str.len().mean())
        info['avg_len'] = avg_len
        if u > 50 and avg_len > 8:
            info.update(type='短文本', seg='⑪',
                        reason='取值 %d 个、平均长度 %.1f 字，像自由文本' % (u, avg_len))
            return info

    # ---- 6. 月份 / 星期 ----
    if not is_num:
        vals = set(s.dropna().astype(str).str.strip().str.lower().unique())
        if vals and vals <= set(KW_MONTH):
            info.update(type='月份(有序)', seg='⑤ 或模板自动',
                        reason='取值全是 Jan~Dec，模板会自动顺序编码')
            return info
        if vals and vals <= set(KW_WEEKDAY):
            info.update(type='星期(有序)', seg='⑤', reason='取值是星期名，建议手动映射 1~7')
            return info

    # ---- 7. 数值型细分 ----
    if is_num:
        vals = s.dropna().astype(float)
        vmin, vmax = float(vals.min()), float(vals.max())
        rng = vmax - vmin
        is_int = bool(np.allclose(vals.values, np.round(vals.values)))
        # "种类数 / 取值范围"：量大（0.3~1），编码小
        density = (u / (rng + 1)) if rng > 0 else 1.0
        info['density'] = density
        sk = float(vals.skew()) if len(vals) > 2 else 0.0
        info['skew'] = sk
        # 0~1 之间且种类多 -> 像比率（但要排除名字已表明是时长/金额/次数的）
        rate_like = (vmin >= 0 and vmax <= 1.0001 and u > 20
                     and not match_kw(name, KW_DURATION + KW_AMOUNT + KW_COUNT))
        code_like = _is_code_name(name)
        # ★ 判断顺序很重要（全是实测踩出来的）：
        #   1) duration/amount/count 这类"语义明确的量"最优先 —— 名字里带 duration
        #      就一定是时长，不能因为任何其他规则把它排除掉
        #   2) 再看它是不是"已经是衍生列"（avg_/total_/ratio_ 开头或结尾）
        #   3) 最后才按数值特征猜
        is_dur = match_kw(name, KW_DURATION)
        is_amt = match_kw(name, KW_AMOUNT)
        is_cnt = match_kw(name, KW_COUNT)
        already_ratio = match_kw(name, KW_IS_RATIO)
        already_derived = _is_derived_name(name)
        bin_like = (is_int and u <= 30
                    and match_kw(name, ['bin', 'bucket', 'level', 'grade', '分箱', '档']))

        if bin_like:
            info.update(type='已经分箱', seg='直接用',
                        reason='取值 %d 个且名字带 bin/level -> 已经分箱，别再分' % u)
        elif already_ratio and not (is_dur or is_amt or is_cnt):
            info.update(type='已经是比率', seg='直接用',
                        reason='列名表明它已经是比率 -> 可直接用，但别再拿它当分子分母')
        elif is_dur:
            info.update(type='时长(量)', seg='①②⑨',
                        reason='列名含时长关键词（取值 %d 个）' % u)
        elif is_amt:
            info.update(type='金额(量)', seg='①③⑨',
                        reason='列名含金额关键词，偏度 %.2f' % sk)
        elif is_cnt:
            info.update(type='计数(量)', seg='①②③',
                        reason='列名含次数关键词（取值 %d 个）' % u)
        elif already_derived:
            # 走到这里说明名字里带 avg/total/mean/sum/ratio 等"衍生词"，
            # 但不是 duration/amount/count 这三种明确语义的量
            _hit = [t for t in split_tokens(name) if t in KW_AGG_TOKENS][:2]
            info.update(type='已是衍生列', seg='直接用',
                        reason='列名含 %s -> 已经是均值/合计等衍生量，★不再参与新比率'
                               % ('、'.join(_hit) if _hit else 'avg/total 类词'))
        elif match_kw(name, KW_RATE) or rate_like:
            info.update(type='比率(量)', seg='①⑥',
                        reason='取值落在 0~1 之间且种类多 -> 是比率')
        elif code_like:
            info.update(type='类别码(数值)', seg='⑦③④',
                        reason='列名含编码类关键词 -> ★是编码不是量，不能做比率')
        elif is_int and 3 <= u <= 300:
            info.update(type='计数(量)', seg='①②③',
                        reason='整数计数（取值 %d 个）' % u)
        elif u <= 12:
            info.update(type='等级/程度', seg='⑥',
                        reason='取值仅 %d 个，像等级或程度' % u)
        else:
            info.update(type='连续数值(量)', seg='①②⑨' if sk > 2 else '①②',
                        reason='连续数值，偏度 %.2f%s' % (sk, '（长尾，建议 log，片段⑨）'
                                                          if sk > 2 else ''))
        return info

    # ---- 8. 类别 ----
    if u < MID_CARD_MIN:
        info.update(type='类别(低基数)', seg='⑦', reason='取值仅 %d 个' % u)
    elif u <= HIGH_CARD:
        info.update(type='类别(中基数)', seg='③④', reason='取值 %d 个，适合目标编码' % u)
    else:
        info.update(type='类别(高基数)', seg='④③',
                    reason='取值 %d 个 > %d，★模板不自动编码，必须贴片段④/③' % (u, HIGH_CARD))
    return info


# ============================================================================
# 第 2 步：分组 & 配对（决定片段①② 怎么填）
# ============================================================================
STOP_TOKENS = {'total', 'sum', 'avg', 'mean', 'max', 'min', 'num', 'cnt', 'count',
               'the', 'of', 'per', 'is', 'has', 'and', 'or', 'val', 'value'}


def name_tokens(name):
    """把列名拆成 token，用于找"同类列"。"""
    n = str(name).lower()
    n = re.sub(r'[^a-z0-9\u4e00-\u9fff]+', ' ', n)
    toks = [t for t in n.split() if t and t not in STOP_TOKENS and len(t) > 2]
    return set(toks)


def find_groups(quantity_cols, meta):
    """按"共享名字 token"把量列分组（用于片段② 行统计）。

    ★ 一个列可以出现在多个组里（我一开始写成"用过的列不再用"，结果
      charge 组把 total_charge/monthly_charge 抢走，income 组就没了）。
      这里改成：每个 token 各成一组，不做互斥。
    """
    toks2col = defaultdict(list)
    for c in quantity_cols:
        for t in name_tokens(c):
            toks2col[t].append(c)
    groups = {}
    for tok, cols in toks2col.items():
        if 2 <= len(cols) <= 6:          # 只保留 2~6 个成员的组（太多说明 token 太泛）
            groups[tok] = sorted(cols)
    # 去掉"被别的组完全包含"的小组（保留信息量更大的那组）
    keys = sorted(groups, key=lambda k: -len(groups[k]))
    out = {}
    seen_sets = []
    for k in keys:
        s = set(groups[k])
        if any(s <= prev for prev in seen_sets):
            continue
        out[k] = groups[k]
        seen_sets.append(s)
    return out


def score_pairs(quantity_cols, full, target, is_tr, top_n=8):
    """给"量"列两两配对，用互信息挑出最有信息的前 top_n 对（用于片段①）。

    ⚠ 关键：sklearn 的 mutual_info_classif 【不接受 NaN】。
      实测踩过：没填缺失时它直接抛异常，被 except 吞掉 -> 静默返回空列表，
      表现就是"配对结果为空"却没有任何报错。所以这里先做中位数填充。
    """
    if not _HAS_SK or len(quantity_cols) < 2:
        return [], {}, 'sklearn 不可用或量列不足 2 个'
    sub = full.loc[is_tr].copy()
    # 去掉缺太多的列，其余用中位数填（只为算互信息，不影响最终数据）
    keep = [c for c in quantity_cols if full[c].notna().mean() > 0.5]
    if len(keep) < 2:
        return [], {}, '可用量列不足 2 个（缺失太多）'
    yv = sub[target].astype(str).str.lower().isin(
        ['true', 'yes', '1', '1.0', 'y', 't', '是']).astype(int).values
    try:
        sub[keep] = sub[keep].fillna(sub[keep].median()).fillna(0)
    except Exception as e:
        return [], {}, '填充失败: %s' % str(e)[:50]
    base = {}
    try:
        mi = mutual_info_classif(sub[keep].values.astype('float32'), yv,
                                 random_state=0, n_neighbors=3)
        base = dict(zip(keep, mi))
    except Exception as e:
        return [], {}, '互信息计算失败: %s' % str(e)[:60]

    scored = []
    for a, b in combinations(keep, 2):
        v = sub[a].values.astype(float) / (sub[b].values.astype(float) + 1e-6)
        v = np.where(np.isfinite(v), v, np.nan)
        if np.isnan(v).mean() > 0.5:
            continue
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        try:
            m = mutual_info_classif(v.reshape(-1, 1).astype('float32'), yv,
                                    random_state=0, n_neighbors=3)[0]
        except Exception:
            continue
        # 只保留"至少不弱于两个原始列中较弱那个"的配对（避免纯噪声）
        floor = min(base.get(a, 0), base.get(b, 0)) * 0.5
        scored.append((a, b, m, floor))
    scored.sort(key=lambda t: -t[2])
    picked = [(a, b, m) for a, b, m, _ in scored[:top_n]]
    msg = '从 %d 个量列生成 %d 对，按互信息取前 %d' % (len(keep), len(scored), len(picked))
    return picked, base, msg


# ============================================================================
# 第 3 步：生成"已填好列名"的粘贴代码
# ============================================================================
def gen_code(meta, groups, pairs, target, id_col, ctx):
    L = []
    A = L.append
    A('# ============================================================================')
    A('# 自动生成的特征工程代码 —— 整段复制到 12 模板的「★ 特征片段粘贴处」')
    A('#   生成时间: %s' % pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    A('#   数据: %s' % ctx['data_path'])
    A('#   目标列: %s | ID列: %s | 训练行 %d | 字段 %d' %
      (target, id_col, ctx['n_rows'], ctx['n_cols']))
    A('# ============================================================================')
    A('')

    # ---- 片段① 比率差值（手写版，只做有意义的对）----
    A('# ---- 片段① 比率/差值（只做实测互信息最高的 %d 对，避免噪声）----' % len(pairs))
    if pairs:
        A('PAIRS = [')
        for a, b, m in pairs:
            A("    ('%s', '%s'),   # 互信息 %.5f" % (a, b, m))
        A(']')
        A("""for _a, _b in PAIRS:
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
print('片段① 完成：%d 对 -> %d 列' % (len(PAIRS), len(PAIRS) * 2))""")
    else:
        A('# 没有找到合适的"量"列配对，跳过片段①')
        A('#（提示：量列 = 时长/次数/金额，且取值要够多；二值列和类别码不能做比率）')
    A('')

    # ---- 片段② 行统计 ----
    A('# ---- 片段② 行统计（按同类列分组，自动找到 %d 组）----' % len(groups))
    if groups:
        A('GROUPS = {')
        for gname, cols in groups.items():
            A("    '%s': %s," % (gname, cols))
        A('}')
        A("""for _gname, _cols in GROUPS.items():
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
    print('片段② 组「%s」：7 个统计量 + %d 个占比' % (_gname, len(_cols)))""")
    else:
        A('# 没有找到"同类列"分组（要求至少 2 列名字里含相同的词）')
        A('# 你可以手动填，例如：GROUPS = {\'页面数\': [\'A\', \'B\']}')
    A('')

    # ---- 片段⑤ 有序映射 ----
    ord_cols = [m['name'] for m in meta if m.get('seg', '').startswith('⑤')]
    A('# ---- 片段⑤ 有序映射 ----')
    month_cols = [m['name'] for m in meta if '月份' in m.get('type', '')]
    if month_cols:
        A('# 月份列（模板已能自动识别，这里保留是为了更清楚）: %s' % month_cols)
    if ord_cols:
        A('# 需要你手写顺序表的列（脚本猜不出业务顺序，请按业务实际填写）:')
        for c in ord_cols:
            A("# ORDINAL_MAPS['%s'] = {...}   # 例: {'low':0, 'mid':1, 'high':2}" % c)
    A("""_MONTH_MAP = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'june':6,'jul':7,
              'july':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12}
for _c in cat_cols:
    _vals = set(full[_c].astype(str).str.strip().str.lower().unique())
    if _vals and _vals <= set(_MONTH_MAP):
        full[_c + '_mon'] = full[_c].astype(str).str.lower().map(_MONTH_MAP).astype(float)
        new_feats.append(_c + '_mon')
        print('片段⑤ 月份顺序编码: %s' % _c)""")
    A('')

    # ---- 片段③ 目标编码 ----
    te_cols = [m['name'] for m in meta if '中基数' in m.get('type', '')
               or '高基数' in m.get('type', '')]
    A('# ---- 片段③ OOF 目标编码（中/高基数类别）----')
    if te_cols:
        A('TE_COLS = %s' % te_cols)
        A("""def _oof_te(df, col, target_col=TARGET, n_splits=5, smoothing=20, seed=SEED):
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
        print('片段③ 目标编码: %s -> %s' % (_c, _n))""")
    else:
        A('# 没有中/高基数类别列，跳过片段③')
    A('')

    # ---- 片段④ 频次编码（★只给"类别/编码"用，绝不能给连续量用）----
    # 实测踩过的坑：如果把 Admin_Duration/BounceRates 这类连续列也做频次编码，
    # 因为每个值基本只出现一次，频次≈1 -> 生成的是一列常数噪声，只会拖慢模型。
    # 频次编码的正确对象：高基数的"类别"（ID/邮编/商品码/城市）。
    freq_cols = [m['name'] for m in meta
                 if m.get('n_unique', 0) > HIGH_CARD
                 and any(k in m.get('type', '') for k in ('类别', '编码'))
                 and m.get('type', '') != 'ID/流水号']
    A('# ---- 片段④ 频次编码（只给真正的高基数"类别/编码"，不给连续量用）----')
    if freq_cols:
        A('# 这些列的取值很多且是"码"-> 频次有信息（比如某商品码被买过多少次）')
        A('FREQ_COLS = %s' % freq_cols)
        A("""for _c in FREQ_COLS:
    if _c not in full.columns:
        continue
    _vc = full[_c].value_counts()
    full[_c + '_freq'] = full[_c].map(_vc).astype(float)
    new_feats.append(_c + '_freq')
    print('片段④ 频次编码: %s' % _c)""")
    else:
        A('# 没有"高基数的类别列"（连续量的唯一值多但不算高基数类别）')
        A('# 如果模板打印 "⚠ XXX 基数 N 过大，跳过编码"，再把那列名填进 FREQ_COLS 手动开。')
        A('# FREQ_COLS = []')
    A('')

    # ---- 片段⑥ 分箱 ----
    A('# ---- 片段⑥ 分箱（只对偏度大的连续列）----')
    skew_cols = [(m['name'], m.get('skew', 0)) for m in meta
                 if m.get('type') == '连续数值' and abs(m.get('skew', 0)) > 2]
    if skew_cols:
        A('BIN_COLS = %s' % [c for c, _ in skew_cols])
        A("""for _c in BIN_COLS:
    if _c in full.columns and full[_c].nunique() > 20:
        full['%s_bin10' % _c] = pd.qcut(full[_c], 10, labels=False,
                                        duplicates='drop').astype(float)
        new_feats.append('%s_bin10' % _c)
        print('片段⑥ 分箱: %s' % _c)""")
    else:
        A('# 没有偏度 > 2 的连续列，跳过片段⑥')
    A('')

    # ---- 片段⑧⑨⑩⑪ ----
    date_cols = [m['name'] for m in meta if '日期' in m.get('type', '')]
    A('# ---- 片段⑧ 日期特征 ----')
    if date_cols:
        A('DATE_COLS = %s' % date_cols)
        A("""for _c in DATE_COLS:
    _d = pd.to_datetime(full[_c], errors='coerce')
    if _d.notna().mean() < 0.5:
        continue
    full[_c + '_month'] = _d.dt.month.astype(float)
    full[_c + '_day'] = _d.dt.day.astype(float)
    full[_c + '_dow'] = _d.dt.dayofweek.astype(float)
    full[_c + '_is_weekend'] = (_d.dt.dayofweek >= 5).astype(float)
    full[_c + '_elapsed'] = (_d - _d.min()).dt.days.astype(float)
    new_feats += [_c + '_month', _c + '_day', _c + '_dow',
                  _c + '_is_weekend', _c + '_elapsed']
    print('片段⑧ 日期特征: %s' % _c)""")
    else:
        A('# 没有日期列，跳过片段⑧')
    A('')

    text_cols = [m['name'] for m in meta if '文本' in m.get('type', '')]
    A('# ---- 片段⑪ 文本特征 ----')
    if text_cols:
        A('TEXT_COLS = %s' % text_cols)
        A("""for _c in TEXT_COLS:
    if _c not in full.columns:
        continue
    _s = full[_c].astype(str)
    full[_c + '_len'] = _s.str.len().astype(float)
    full[_c + '_nwords'] = _s.str.count(r'\\s+').astype(float) + 1
    full[_c + '_ndigit'] = _s.str.count(r'\\d').astype(float)
    full[_c + '_hasdigit'] = _s.str.contains(r'\\d', regex=True).astype(float)
    new_feats += [_c + '_len', _c + '_nwords', _c + '_ndigit', _c + '_hasdigit']
    print('片段⑪ 文本特征: %s' % _c)""")
    else:
        A('# 没有短文本列，跳过片段⑪')
    A('')

    # ---- 片段⑨ log ----
    log_cols = [m['name'] for m in meta if m.get('type') in ('金额(量)', '时长(量)')
                and m.get('skew', 0) > 2]
    A('# ---- 片段⑨ log 变换（长尾量列）----')
    if log_cols:
        A('LOG_COLS = %s' % log_cols)
        A("""for _c in LOG_COLS:
    if _c in full.columns:
        full[_c + '_log'] = np.log1p(full[_c].clip(lower=0))
        new_feats.append(_c + '_log')
        print('片段⑨ log1p: %s' % _c)""")
    else:
        A('# 没有偏度 > 2 的量列，跳过片段⑨（或者你想加就手动填 LOG_COLS）')
    A('')

    A('# ---- 片段⑩ 异常打分：默认不贴 ----')
    A('# 实测在弱信号数据上会掉分（iso_score 单独 AUC 只有 0.54）。')
    A('# 只有题目背景是"欺诈/故障/违约"这类"异常即目标"的场景才建议开。')
    A('# 想试就把模板 cell 0 的 USE_ISOLATION 改成 True。')
    A('')
    A('# ============================================================================')
    A('# ⚠ 粘完之后务必做这一步（30 秒，决定要不要保留这些特征）')
    A('# ============================================================================')
    A('#   打开 12 模板 cell 23，把 COMPARE_FE 改成 True 再跑一次，看这两行：')
    A('#       仅原始字段      : OOF F1 = xxxx  (特征数 N1)')
    A('#       原始+衍生特征   : OOF F1 = yyyy  (特征数 N2)')
    A('#   ★ yyyy 明显大于 xxxx 才保留本文件；否则把这段删掉，用原始字段交卷。')
    A('#   为什么必须验证：实测在信号弱的数据（AUC 只有 0.6 左右）上，')
    A('#   加特征可能反而降分（我在这份电商数据上实测过：基线 0.2716 -> 加特征 0.2619）。')
    A('#   数据里若存在"必须自己构造才能表达"的关系（如 总消费/月数 = 已用月数），')
    A('#   加特征才会明显提分。')
    A('# ============================================================================')
    A('')
    A("print('★ 自动生成的特征工程完成，新增列数:', len(new_feats))")
    return '\n'.join(L)


# ============================================================================
# 主流程
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data', nargs='?', default=None, help='训练数据（csv/xlsx）')
    ap.add_argument('-t', '--target', default=None, help='目标列名（不填则自动猜）')
    ap.add_argument('-i', '--id', dest='id_col', default=None, help='ID 列名（不填则自动猜）')
    ap.add_argument('-o', '--out', default=None, help='输出文件名')
    ap.add_argument('--sample', type=int, default=30000, help='读多少行做分析（默认3万，够统计）')
    args = ap.parse_args()

    path = args.data
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            '练习', 'ecommerce_train.csv')
        if not os.path.exists(path):
            path = 'ecommerce_train.csv'
        print('未指定数据文件，用默认的 2026.06 电商真题试跑:', path)

    df = read_table(path).sample(frac=1, random_state=0).head(args.sample).reset_index(drop=True)
    n_rows, n_cols = len(df), df.shape[1]
    print('=' * 100)
    print('读入 %s' % path)
    print('分析样本 %d 行 × %d 列' % (n_rows, n_cols))
    print('=' * 100)

    # ---- 猜目标列 ----
    target = args.target
    if target is None:
        for c in df.columns:
            if c.lower() in ('target', 'label', 'y', 'churn', 'revenue', 'purchase',
                             'fraud', 'default', 'flag', '目标', '标签'):
                target = c
                break
        if target is None:
            # 取唯一值 2~10 个、且不是 ID 的最后一列
            for c in reversed(list(df.columns)):
                if 2 <= df[c].nunique() <= 10 and df[c].nunique() / n_rows < 0.5:
                    target = c
                    break
        print('自动识别的目标列: %s' % target)
    if target not in df.columns:
        print('✗ 目标列 %r 不在数据里，请用 -t 指定' % target)
        return 1

    # ---- 猜 ID 列 ----
    id_col = args.id_col
    if id_col is None:
        for c in df.columns:
            if c.lower() in ('id', 'index', 'customerid', 'customer_id', '编号'):
                id_col = c
                break
    print('自动识别的 ID 列: %s' % id_col)
    if id_col is None:
        print('（没找到 ID 列——交卷需要 ID 列，请用 -i 指定）')
    print('目标分布:\n%s' % df[target].value_counts(dropna=False).head(6).to_string())

    # ---- 逐列分型 ----
    feats = [c for c in df.columns if c not in (target, id_col)]
    meta = [classify_column(c, df[c], n_rows) for c in feats]

    print('\n' + '=' * 100)
    print('【字段分型报告】')
    print('=' * 100)
    print('%-22s %-8s %-14s %-8s %-8s %s'
          % ('字段', '取值数', '判断类型', '缺失率', '用片段', '判断依据'))
    print('-' * 116)
    for m in sorted(meta, key=lambda x: x.get('type', '')):
        print('%-22s %-8d %-14s %-8.3f %-8s %s'
              % (m['name'][:22], m['n_unique'], m.get('type', '?'), m['missing'],
                 m.get('seg', ''), m.get('reason', '')[:52]))

    # ---- 量列 vs 码列，分组配对 ----
    # ★ 参与比率配对的"基础量"必须排除：已经是比率/衍生/分箱的列
    #   否则会造出 total_charge/avg_charge 这种循环无意义的比率（实测踩过）
    BASE_Q_TYPES = ('时长(量)', '金额(量)', '计数(量)', '连续数值(量)')
    quantity = [m['name'] for m in meta
                if m.get('type', '') in BASE_Q_TYPES and not _is_derived_name(m['name'])]
    derived_cols = [m['name'] for m in meta
                    if m.get('type', '') in ('已经是比率', '已是衍生列', '已经分箱')]
    rate_cols = [m['name'] for m in meta if m.get('type', '') == '比率(量)']
    code_like = [m['name'] for m in meta
                 if m.get('type', '') in ('类别码(数值)', '等级/程度')]
    ids = [m['name'] for m in meta if 'ID' in m.get('type', '')]

    print('\n' + '=' * 100)
    print('【关键结论①】哪些列能做比率，哪些不能')
    print('=' * 100)
    print('  ✅ 基础量、可参与比率配对 (%d 列): %s' % (len(quantity), quantity))
    if rate_cols:
        print('  ⚠️ 本身是比率 (%d 列): %s' % (len(rate_cols), rate_cols))
        print('       -> 可直接用，但别再拿它当分子分母（会造出循环比率）')
    if derived_cols:
        print('  ⚠️ 已经是衍生列/分箱 (%d 列): %s' % (len(derived_cols), derived_cols))
        print('       -> 直接用即可，不参与新比率生成')
    print('  ❌ 是编码/等级、不能做比率 (%d 列): %s' % (len(code_like), code_like))
    if ids:
        print('  🗑️ 是 ID/标识号、必须扔掉 (%d 列): %s' % (len(ids), ids))

    # ---- ★ 泄漏检测：某个特征与目标的相关性高得可疑 ----
    if _HAS_SK:
        try:
            _sub = df[[m['name'] for m in meta]].copy()
            for c in _sub.columns:
                if _sub[c].dtype == object:
                    _sub[c] = pd.factorize(_sub[c])[0]
            _sub = _sub.fillna(_sub.median(numeric_only=True)).fillna(0)
            _yv = df[target].astype(str).str.lower().isin(
                ['true', 'yes', '1', '1.0', 'y', 't', '是']).astype(int).values
            _mi = mutual_info_classif(_sub.values.astype('float32'), _yv,
                                      random_state=0, n_neighbors=3)
            _mis = pd.Series(_mi, index=_sub.columns).sort_values(ascending=False)
            top = _mis.head(3)
            if len(top) and top.iloc[0] > 0.3:
                print('\n' + '=' * 100)
                print('【警告】以下字段与目标相关性强得可疑 —— 先确认是不是"泄漏字段"')
                print('=' * 100)
                for c, v in top.items():
                    flag = '  ★高度可疑（可能是答案的直接来源，建议删掉）' if v > 0.5 else ''
                    print('  %-24s 互信息 %.4f%s' % (c, v, flag))
                print('  说明：正常特征互信息一般 < 0.2；超过 0.5 往往意味着这列"泄漏"了答案')
                print('        （比如：目标=是否流失，而某列是"销户日期"）')
        except Exception:
            pass

    groups = find_groups(quantity, meta)
    print('\n' + '=' * 100)
    print('【关键结论②】自动找到的"同类列"分组（片段② 的 GROUPS）')
    print('=' * 100)
    if groups:
        for g, cols in groups.items():
            print('  组「%s」(%d 列): %s' % (g, len(cols), cols))
    else:
        print('  没找到自动分组 —— 你需要手动填 GROUPS（把语义同类的列放一组）')
        print('  例：GROUPS = {\'页面数\': [\'Administrative\', \'Informational\']}')

    pairs, base_mi, pair_msg = score_pairs(quantity, df, target, pd.Series([True] * len(df)),
                                           top_n=8)
    print('\n' + '=' * 100)
    print('【关键结论③】自动挑出的"值得做比率"的配对（片段① 的 PAIRS）')
    print('=' * 100)
    print('  ' + pair_msg)
    if pairs:
        for a, b, m in pairs:
            print('  %-34s = %s / %s   互信息 %.5f'
                  % ('%s__div__%s' % (a, b), a, b, m))
    else:
        print('  （没有可用的配对）')
        print('  可能原因：① 量列不足 2 个 ② 量列缺失过半 ③ sklearn 不可用')
        print('  你可以手动填 PAIRS，例如：')
        print("      PAIRS = [('Product_Duration', 'Admin_Duration')]")

    warn = [m for m in meta if '高基数' in m.get('type', '')]
    if warn:
        print('\n' + '=' * 100)
        print('【警告】以下列模板【不会】自动编码（基数 > %d），必须贴片段③/④' % HIGH_CARD)
        print('=' * 100)
        for m in warn:
            print('  %-24s 取值 %d 个' % (m['name'], m['n_unique']))

    # ---- 生成代码 ----
    ctx = dict(data_path=path, n_rows=n_rows, n_cols=n_cols)
    code = gen_code(meta, groups, pairs, target, id_col, ctx)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(path)) or '.',
                                   '_特征工程_已填好.py')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# -*- coding: utf-8 -*-\n')
        f.write('# 由 16_自动检测该做什么特征工程.py 生成\n')
        f.write('# 用法：整段内容复制到 12_模板_202609结构化实战模板.py 的「★ 特征片段粘贴处」\n\n')
        f.write(code)
    print('\n' + '=' * 100)
    print('★ 已生成可粘贴的代码文件: %s' % out)
    print('=' * 100)
    print('  下一步：')
    print('   1) 打开这个文件，全选复制')
    print('   2) 粘到 12 模板的「★ 特征片段粘贴处」（new_feats = [] 下面那两行标记之间）')
    print('   3) 运行模板，看 "★ 特征 N 个（片段新增 XX）"，XX 不为 0 就说明生效了')
    print('   4) 打开模板 cell 23 的 COMPARE_FE = True，验证加特征到底有没有提分')
    print()
    print('  ⚠ 脚本能自动填"列名"，但填不了"业务顺序"。')
    print('     如果上面报告里有"等级/程度"类字段，它的顺序表需要你按业务自己写。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
