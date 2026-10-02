# ============================================================================
# 12_模板_202609结构化实战模板.py —— 2026.09 考场主模板（精简 · 高效版）
# ============================================================================
# 题型（PPT 已公布）：结构化表格 · 二分类 · 训练约 24000 / 测试约 6000
#   原始字段约 30 个，要求特征工程产生新字段；目标轻微不平衡（正类约 35%）；
#   要参数优化 + 交叉验证选模型（集成较优）+ 门坎值优化
#
# 评分档位（PPT 原文）：<0.40 预处理不足 | 0.40~0.49 基础预处理 |
#   0.50~0.55 特征工程有效（保底）| 0.56~0.59 强特征工程+调参（目标）| >=0.60 优秀
#
# 提交（错一个字 = 0 分）：文件名 考生姓名_results.csv，两列 id / Predicted_Results，
#   取值照 docx 范例的大小写（TRUE/FALSE 或 Yes/No）；代码结果必须和 CSV 一致
#
# ---------------------------------------------------------------------------
# 性能设计（实测数据，都是踩坑后加的）
# ---------------------------------------------------------------------------
#   【最重要的三个速度修复，实测总耗时 223s -> 29s】
#   1) RF 树数按数据量缩放 + 绝不给它设 n_estimators=2000：
#      曾经因为"早停轮数复用"把 RF 从 100 棵变成 2000 棵（慢 20 倍）；
#      又因为给 6 万行数据用 300 棵完全生长的树，单折要 70 秒。
#      现在：≤3 万行 300 棵、更大 100 棵、>15 万行 50 棵。RF 5 折从 210s -> 14s。
#   2) 门坎搜索用"排序后一次扫描"O(n log n)，不是对每个门坎重算 F1（那是 O(n²)，
#      28 万行时要上亿次运算，看起来就像卡死）。
#   3) 线程数自动取 min(8, 逻辑核数/2)。实测 i7-14650HX（16 核）28 万行 LGBM：
#      1线程 13.0s / 4线程 5.7s / 8线程 3.2s（最快）/ 12线程 7.2s / 16线程 8.2s
#      -> 开满反而更慢（超订 + 内存带宽争抢），所以别用 -1。
#
#   【早停的坑（lightgbm 4.6 + pandas 实测）】
#     回调会正常提前停止（103 轮就停），但它报的 best_iteration_ 是错的（报 3）。
#     所以模板只借用"提前停止"省时间，训练轮数一律自己从平滑后的验证曲线算，
#     并设最小轮数下限（否则小验证集的噪声会让模型在第 20 轮就停 = 欠拟合）。
#
#   【模型耗时排序（2.4 万行 5 折，2026.06 电商数据实测）】
#     LGBM 6s < CatBoost 8s < RF 33s < XGB 47s   -> 总共约 100s
#   【GPU】实测 RTX 4060 / 28 万行：XGBoost CPU 2.4s vs CUDA 5.8s（GPU 更慢，别开）；
#     CatBoost CPU 12.1s vs GPU 5.1s（GPU 快 2.4 倍）。pip 版 lightgbm 没有 GPU。
# ============================================================================

# ============================================================================
# 配置区（只改这里）
# ============================================================================
import warnings
import os
import sys
import time

warnings.filterwarnings('ignore')

TRAIN_PATH = 'ecommerce_train.csv'
TEST_PATH = 'ecommerce_test.csv'
TARGET = 'Revenue'
ID_COL = 'ID'

# ---- 正类怎么定（练习别的数据集时改这里）--------------------------------
#  'auto'  = 自动：数值列取大的那类；文本列按 TRUE/Yes/流失 这类常见正类词判断
#  字符串  = 直接指定，例如 'Yes' / 'TRUE' / '1' / '流失'
#  想评负类（练习用）就填负类那个值，例如 'No' / '0'
#  考试提醒：202609 评的是"正类的 F1"，照 docx 提交范例填最稳
POSITIVE_CLASS = 'True'

# ---- 提交格式（照 docx「考生姓名_results.csv 范例」那段抄）--------------
SUB_PATH = '考生姓名_results.csv'
SUB_COL = 'Predicted_Results'      # 预测列名
# ★★ 提交文件里【正负类写成什么】—— 这一项错了就是 0 分，务必和 docx 范例逐字对。
#    'auto'      = 按【原始目标值】的字面推导（推荐）：
#                    目标是 0/1 这种数字  -> 输出 '1' / '0'
#                    目标是 True/False    -> 输出 'TRUE' / 'FALSE'
#                    目标是 Yes/No        -> 输出 'Yes' / 'No'
#    'TRUE_FALSE'= 强制输出 TRUE / FALSE
#    'YES_NO'    = 强制输出 Yes / No
#    'NUM'       = 强制输出 1 / 0
#    '原样'      = 用目标列里那两个原始写法（极少用）
#  历年真题的实际要求（我逐年查过 docx，差别真的很大）：
#    2026.06 电商 -> TRUE / FALSE     2026.03 电信 -> Yes / No
#    2024.12 欺诈 -> 0 / 1  ★注意是数字！   2025.12 故障 -> normal/cf1/... （多分类）
#  ⇒ 所以绝不能写死 TRUE/FALSE，必须每次照 docx 范例确认一遍。
SUB_FORMAT = 'auto'

# ---- 性能开关（大表卡就调这几个）----------------------------------------
# 线程数：'auto' = 自动取 min(8, 逻辑核数//2)。
#   实测（i7-14650HX 16 核，28 万行 LightGBM 单次训练）：1线程 13.0s / 4线程 5.7s /
#   8线程 3.2s（最快）/ 12线程 7.2s / 16线程 8.2s。
#   => 开满反而更慢（超订 + 内存带宽争抢），所以别用 -1，也别只用 4。
#   机器上还有别的任务在抢 CPU 时，往下调（比如 4）；独占机器可试 8~10。
N_JOBS = 'auto'
MAX_FIT_ROWS = 300000  # 训练行数超过它才做分层抽样（24000 行的考题不会触发）
ES_FRAC = 0.10         # 早停验证集比例（10% 足够稳；太小曲线噪声大，会误停在几十轮）
DEBUG_ES = False       # True = 打印早停探针（诊断用；平时关掉，输出更干净）
REUSE_BEST_ITER = True # 早停轮数算一次、各折复用（省 30%~50% 训练时间）
ROW_CAP = 0
                       # OOF 仍只在抽到的样本上算指标；考试 2.4 万行请保持 0

# ---- 其它开关 ------------------------------------------------------------
QUICK = False
USE_CATBOOST = True

# GPU 开关（'auto' / True / False）。只影响 CatBoost，其余模型一律 CPU。
#   'auto' = 按数据量自动决定：训练行数 > GPU_MIN_ROWS 时才给 CatBoost 开 GPU
#   为什么不做成"全开"：实测 GPU 在小数据上是【负优化】，而且不是所有模型都支持
#     · LightGBM：pip 版没编译 GPU，用 device='gpu' 会直接报
#       "GPU Tree Learner was not enabled in this build"；写 device='cuda' 不报错
#       但那是静默回退到 CPU（实测 CPU 0.2s vs "cuda" 1.3s），别被假象骗了
#     · RandomForest：sklearn 版本没有 GPU
#     · XGBoost：实测 GPU 在两个数据量下都更慢（见下表）-> 永远别开
#     · CatBoost：唯一有意义的，但要看数据量（见下表）
#   【本机实测：RTX 4060 Laptop 8GB / 18 个特征 / 都带早停】
#     数据量      CatBoost CPU   CatBoost GPU   结论
#     24,000行    8.0 秒         32.8 秒        CPU 快 4.1 倍  ← 202609 考场就是这个量级
#     120,000行   6.6 秒         6.3 秒         打平
#     280,000行   22.7 秒        6.6 秒         GPU 快 3.4 倍
#     XGBoost     9.0 / 37.9 秒  20.2 / 23.6 秒 GPU 始终更慢
#   显存占用很小（28 万行时 CatBoost GPU 只多占 107 MiB），所以瓶颈不是显存，
#   而是 GPU 的启动 + 数据搬运开销在小数据上收不回来。
USE_GPU = 'auto'
GPU_MIN_ROWS = 100000  # 'auto' 的门槛：训练行数超过它才给 CatBoost 开 GPU
USE_ISOLATION = False
CLIP_Q = 0.001         # 数值缩尾分位（0.001 = 0.1% / 99.9%）
SEED = 42

# ============================================================================
# 依赖（两环境实测：base python3.13 + xgboost3.2；pytorch python3.9 + xgboost2.0.3）
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_es
from xgboost import XGBClassifier
import lightgbm as lgb
import xgboost as _xgb

_XGB_MAJOR = int(str(_xgb.__version__).split('.')[0])
try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False

# 线程数自动探测（N_JOBS='auto' 时）：取 min(8, 逻辑核数 // 2)
if str(N_JOBS).lower() == 'auto':
    _lc = os.cpu_count() or 4
    N_JOBS = max(2, min(8, _lc // 2))
else:
    N_JOBS = int(N_JOBS)

P = lambda *a: print(*a, flush=True)
P('python %s | lightgbm %s | xgboost %s(major %d) | catboost %s | 线程 %d'
  % (sys.version.split()[0], lgb.__version__, _xgb.__version__, _XGB_MAJOR,
     'OK' if _HAS_CATBOOST else '未装(自动跳过)', N_JOBS))


def tick(msg):
    """分阶段计时：卡住时一眼看出是哪一步慢（大表排查用）。"""
    P('[%6.1fs] %s' % (time.time() - T0, msg))


T0 = time.time()


# ============================================================================
# 1. 读数据
# ============================================================================
def clean_columns(df):
    """列名去首尾空格；字符串列去首尾空格（' no' 与 'no' 会裂成两类，很坑）。

    真题实例：201906 的 international_plan 存的是 ' no'/' yes'；
    201812 的列名自带引号（"'Product using score'"）且 ' Estimated salary' 前有空格。
    """
    df.columns = [str(c).strip().strip("'\"") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def _read_one(path, **kw):
    """单个文件的读取（编码兜底 + 脏数据哨兵 + 列名/空格清理）。"""
    na = ['?', '??', '.', '-', '--', 'NA', 'N/A', 'n/a', 'null', 'NULL', 'None',
          'none', 'nan', 'NaN', 'unknown', 'Unknown', '未知', ' ']
    kw = dict(kw)
    kw['na_values'] = na
    if str(path).lower().endswith(('.xlsx', '.xls')):
        return clean_columns(pd.read_excel(path, **kw))
    for enc in ['utf-8-sig', 'gb18030', 'latin-1']:
        try:
            return clean_columns(pd.read_csv(path, encoding=enc, **kw))
        except (UnicodeDecodeError, UnicodeError):
            continue
    return clean_columns(pd.read_csv(path, encoding='latin-1', **kw))


def read_table(path, id_col=None):
    """csv/xlsx/目录都能读，编码自动兜底（utf-8-sig -> gb18030 -> latin-1）。

    脏数据哨兵（? . - -- 空串 等）统一按缺失值处理，否则整列会被当成字符串类别。

    ★★ ID 列必须按【字符串】读（实测踩过的 0 分坑）：
       如果 ID 是 '00000' / '00001' 这种【带前导零的纯数字字符串】，
       pandas 会把它自动推断成整数 0 / 1，前导零当场丢掉。
       提交文件里就变成 0/1/2...，和标准答案的 ID **一个都对不上** —— 直接 0 分。
       所以这里对 ID 列强制 dtype=str。（201912 的 ID 是普通整数，不受影响。）
    """
    kw = {'dtype': {id_col: str}} if id_col else {}
    if os.path.isdir(path):
        dfs = []
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.lower().endswith(('.xlsx', '.xls', '.csv')):
                    dfs.append(_read_one(os.path.join(root, f), **kw))
        return pd.concat(dfs, ignore_index=True)
    return _read_one(path, **kw)


# ★ ID 列名先纠偏，再读数据 —— 因为读的时候就要按名字把 ID 当字符串读。
#   （只读表头，不读全表：大表下这一步几乎不花时间）
_head_cols = []
for _enc in ['utf-8-sig', 'gb18030', 'latin-1']:
    try:
        _head_cols = list(pd.read_csv(TRAIN_PATH, encoding=_enc, nrows=0).columns)
        break
    except (UnicodeDecodeError, UnicodeError, OSError):
        continue
if ID_COL not in _head_cols:
    for c in ['id', 'ID', 'Id', 'customerID', '用户ID']:
        if c in _head_cols:
            ID_COL = c
            P('  （ID 列名自动纠偏为 %r）' % c)
            break

train, test = read_table(TRAIN_PATH, ID_COL), read_table(TEST_PATH, ID_COL)
if ID_COL not in train.columns:                      # 还没找到就用 train 里的候选名
    for c in ['id', 'ID', 'Id', 'customerID', '用户ID']:
        if c in train.columns:
            ID_COL = c
            break
P('\n★ train %s | test %s | ID列=%s | 目标列=%s'
  % (train.shape, test.shape, ID_COL, TARGET))
P('目标分布:\n' + train[TARGET].value_counts(dropna=False).head(5).to_string())


# ============================================================================
# 2. 目标列 -> 0/1（正类 = 评分那一类，可指定）
# ============================================================================
def norm_label(v):
    """统一标签写法：'1.0'->'1'，'TRUE'->'true'（便于跨写法比较）。"""
    s = str(v).strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError):
        return s.lower()


def fmt_label(v):
    """把一个原始标签值规范成"提交文件该写的字样"。

    ★ 为什么不能写死 TRUE/FALSE（逐年查 docx 发现的真坑）：
      2026.06 电商题的范例是 TRUE/FALSE，
      但 2024.12 欺诈题的范例是【数字 0/1】，
      2026.03 电信题是 Yes/No。写死一种，另外两种就是"格式错 = 0 分"。

    规则：原始值是 0/1 这类数字 -> 保持数字；是 True/False -> 大写；
          是 Yes/No -> 首字母大写；其余原样。
    """
    s = str(v).strip()
    if s.lower() in ('true', 'false'):
        return s.upper()
    if s.lower() in ('yes', 'no'):
        return s.capitalize()
    # ★ 数字原样返回（不转 TRUE/FALSE）—— 202412 要的就是 0/1
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else s
    except (TypeError, ValueError):
        pass
    return s


def resolve_sub_format(pos_raw, neg_raw, mode=None):
    """决定提交文件里正/负类到底写什么，返回 (正类字样, 负类字样, 说明)。

    mode 取 SUB_FORMAT：'auto' / 'TRUE_FALSE' / 'YES_NO' / 'NUM' / '原样'
    """
    mode = (mode or 'auto')
    m = str(mode).strip().upper()
    if m in ('TRUE_FALSE', 'TRUEFALSE', 'BOOL'):
        return 'TRUE', 'FALSE', "强制 TRUE/FALSE（配置里指定的）"
    if m in ('YES_NO', 'YESNO'):
        return 'Yes', 'No', "强制 Yes/No（配置里指定的）"
    if m in ('NUM', '01', '0/1'):
        return '1', '0', "强制 1/0（配置里指定的）"
    if m in ('原样', 'RAW', 'KEEP'):
        return str(pos_raw), str(neg_raw), "原样沿用目标列的写法（配置里指定的）"
    # ---- auto：按原始目标值的字面推导 ----
    p, n = fmt_label(pos_raw), fmt_label(neg_raw)
    if {p.lower(), n.lower()} == {'0', '1'}:
        return p, n, "自动判定：原始目标是数字 0/1 -> 写 0/1（202412 欺诈题就是这种）"
    if {p.lower(), n.lower()} == {'true', 'false'}:
        return 'TRUE', 'FALSE', "自动判定：原始目标是布尔 -> 写 TRUE/FALSE（2026.06 电商题是这种）"
    if {p.lower(), n.lower()} == {'yes', 'no'}:
        return 'Yes', 'No', "自动判定：原始目标是 Yes/No -> 写 Yes/No（2026.03 电信题是这种）"
    return p, n, "自动判定：沿用目标列的原始写法 %r / %r" % (p, n)


def binarize(series, positive='auto'):
    """返回 (0/1 序列含 NaN, 正类原值, 负类原值)。positive 可为 'auto' 或 'Yes'/'1' 等。"""
    s = series.copy()
    if s.dtype == object or str(s.dtype) in ('category', 'string', 'bool', 'boolean'):
        s = s.astype(str).str.strip().replace(['nan', 'NaN', 'None', 'NULL', '', '?', '.'], np.nan)
    vals = pd.Series(s.dropna().unique())
    if len(vals) != 2:
        raise ValueError('目标列不是二分类（取值 %s），请改走多分类模板' % list(vals[:6]))
    a, b = vals.iloc[0], vals.iloc[1]
    if positive == 'auto':
        na, nb = pd.to_numeric(pd.Series([a, b]), errors='coerce')
        if pd.notna(na) and pd.notna(nb):
            pos = a if na > nb else b                    # 数值型：取大的那类
        else:
            pos = None
            for w in ['true', 'yes', 'y', '1', 'high', 'positive', '流失', '是']:
                if norm_label(a) == w:
                    pos = a; break
                if norm_label(b) == w:
                    pos = b; break
            if pos is None:
                pos = b
    else:
        hit = [v for v in vals if norm_label(v) == norm_label(positive)]
        if not hit:
            raise ValueError('POSITIVE_CLASS=%r 在目标列里找不到，实际取值: %s'
                             % (positive, list(vals)))
        pos = hit[0]
    neg = b if norm_label(pos) == norm_label(a) else a
    y = s.map(lambda v: np.nan if pd.isna(v)
              else (1.0 if norm_label(v) == norm_label(pos) else 0.0)).astype(float)
    return y, pos, neg


y_bin, POS_RAW, NEG_RAW = binarize(train[TARGET], POSITIVE_CLASS)
train[TARGET] = y_bin
train = train.dropna(subset=[TARGET]).copy()
SUB_POS, SUB_NEG, _fmt_why = resolve_sub_format(POS_RAW, NEG_RAW, SUB_FORMAT)
P('\n★ 正类 = %r -> 输出 %r（%.1f%%）| 负类 = %r -> 输出 %r'
  % (POS_RAW, SUB_POS, 100 * train[TARGET].mean(), NEG_RAW, SUB_NEG))
P('  提交取值格式: %s' % _fmt_why)
P('  ⚠ 请拿这个和 docx 的「考生姓名_results.csv 范例」逐字对一遍：'
  '历年真题要求不一样（2026.06=TRUE/FALSE，2026.03=Yes/No，2024.12=0/1）。')
P('    不对就改 cell 0 的 SUB_FORMAT（\'TRUE_FALSE\' / \'YES_NO\' / \'NUM\' / \'原样\'）')


# ============================================================================
# 3. 清洗：错误值 / 空值 / 离群值
# ============================================================================
test[TARGET] = np.nan
full = pd.concat([train, test], axis=0, ignore_index=True)   # 合并清洗，保证两边列数一致
full['__is_train'] = full[TARGET].notna().astype(int)
is_tr = full['__is_train'] == 1

# 3.1 抢救被读成文本的数值列（金额混了 $ , % 空格）
for c in full.columns:
    if c in (TARGET, ID_COL, '__is_train') or full[c].dtype != object:
        continue
    s = full[c].astype(str).str.strip().str.replace(r'[,$¥￥%\s]', '', regex=True)
    s = s.replace(['', 'nan', 'None', 'NULL', '?', '.'], np.nan)
    conv = pd.to_numeric(s, errors='coerce')
    if full[c].notna().sum() and (conv.notna() & full[c].notna()).sum() / full[c].notna().sum() >= 0.9:
        full[c] = conv

# 3.1b 日期列 -> 数值特征（★原来模板完全没有这一步，实测发现的缺口）
#   真实数据里日期列很常见，而【日期字符串是没法直接进模型的】：
#   如果不管它，它会被当成"高基数类别列"然后被跳过编码（实测：'⚠ 基数过大，跳过编码'），
#   等于白丢一列。这里自动转成 月/日/星期/小时/是否周末 这些模型能用的数字。
#   （16 检测脚本也会生成日期特征代码，但那是要手动粘的；模板自己兜住更稳）
DATE_PARTS = ('month', 'day', 'dow', 'hour', 'is_weekend')
for c in list(full.columns):
    if c in (TARGET, ID_COL, '__is_train') or full[c].dtype != object:
        continue
    raw = full[c].astype(str).str.strip()
    if raw.replace(['', 'nan', 'None', 'NULL'], np.nan).notna().mean() < 0.5:
        continue
    try:
        dt = pd.to_datetime(raw, errors='coerce')
    except Exception:
        continue
    if dt.notna().mean() >= 0.8:                 # 能解析八成以上，就当日期列
        full[c + '_month'] = dt.dt.month.astype(float)
        full[c + '_day'] = dt.dt.day.astype(float)
        full[c + '_dow'] = dt.dt.dayofweek.astype(float)
        full[c + '_hour'] = dt.dt.hour.astype(float)
        full[c + '_is_weekend'] = (dt.dt.dayofweek >= 5).astype(float)
        # _elapsed = 距最早日期多少天。"第几天"往往比"几月几号"更接近业务含义
        #   （例如订单是第几天下的、账号是第几天注册的），且天然单调。
        full[c + '_elapsed'] = (dt - dt.min()).dt.days.astype(float)
        # 原始列保留成数值时间戳（供后续片段算差值），但不再当类别
        full[c + '_ts'] = dt.astype('int64', errors='ignore') / 10 ** 9
        full = full.drop(columns=[c])            # 原字符串列扔掉（高基数、没法编码）
        P('  日期列 %s -> 新增 7 个日期特征（month/day/dow/hour/is_weekend/elapsed/ts）' % c)

# 3.2 文本列去空格，空白当缺失
for c in full.columns:
    if full[c].dtype == object and c != ID_COL:
        full[c] = full[c].astype(str).str.strip()
        full.loc[full[c].isin(['', 'nan', 'None', 'NULL', '?', '.']), c] = np.nan

num_cols = [c for c in full.columns if c not in (TARGET, ID_COL, '__is_train')
            and pd.api.types.is_numeric_dtype(full[c])]
cat_cols = [c for c in full.columns if c not in (TARGET, ID_COL, '__is_train')
            and not pd.api.types.is_numeric_dtype(full[c])]
for c in num_cols:                                   # 布尔列转 0/1（否则分位数报错）
    if str(full[c].dtype) in ('bool', 'boolean'):
        full[c] = full[c].astype(float)

# 3.3 空值：缺失指示特征 + 训练集中位数/新类别填充
for c in list(num_cols) + list(cat_cols):
    if 0.005 < full[c].isna().mean() < 0.40:
        full[c + '_isna'] = full[c].isna().astype(int)
for c in num_cols:
    full[c] = pd.to_numeric(full[c], errors='coerce')
    full[c] = full[c].fillna(full.loc[is_tr, c].median()).fillna(0)
for c in cat_cols:
    full[c] = full[c].fillna('__MISSING__').astype(str)

# 3.4 离群值：按训练集分位缩尾
for c in num_cols:
    lo, hi = full.loc[is_tr, c].quantile(CLIP_Q), full.loc[is_tr, c].quantile(1 - CLIP_Q)
    if lo < hi:
        full[c] = full[c].clip(lo, hi)

if USE_ISOLATION:
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=N_JOBS)
    iso.fit(full.loc[is_tr, num_cols])
    full['__iso_score'] = -iso.score_samples(full[num_cols])
    num_cols = num_cols + ['__iso_score']
P('\n清洗完成：数值列 %d | 类别列 %d | 剩余缺失 %d'
  % (len(num_cols), len(cat_cols), int(full.isna().sum().sum())))
tick('清洗完成（错误值/空值/离群值）')


# ============================================================================
# 4. ★ 特征片段粘贴处（从 13_特征工程片段库_202609.py 复制到这里）
# ============================================================================
# 已准备好的变量（片段库里每个片段都直接用）：
#   full          train+test 合并大表（清洗已完成）
#   TARGET / ID_COL / __is_train
#   num_cols      数值列名列表          cat_cols   类别列名列表
#   new_feats     新造的【数值】特征名（片段往这里 append，会自动进模型）
#   keep_feats    直接进模型的编码列（目标编码/有序编码这类往这里 append）
#
# 挑法（按性价比）：必贴【① 比率差值】；推荐【② 行统计】【③ OOF目标编码】【⑤ 有序映射】
# 大表提醒：片段① 的 PAIR_TOP_N 别调太大（12~15 够用），特征太多会拖慢训练
# ============================================================================
new_feats = []
keep_feats = []

# --- 粘贴处开始（把片段贴在这一行下面）-------------------------------------
# -*- coding: utf-8 -*-

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
# --- 粘贴处结束 -------------------------------------------------------------


# ============================================================================
# 5. 编码 + 拼特征矩阵
# ============================================================================
# ★ 月份编码不再由模板做，改由「片段⑤」负责（避免重复生成 Month_mon 又去重丢掉）。
#   原因：片段库/自动生成代码里已经有月份顺序编码，模板再做一遍是白费功夫。
#   风险与兜底：如果既没贴片段⑤、月份列又会落进 LabelEncoder，
#   会被按字母序乱编（Apr<Aug<Dec<…），下面循环末尾会自动检测并警告 + 自动补一个。
_MONTH_MAP = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'june': 6,
              'jul': 7, 'july': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10,
              'nov': 11, 'dec': 12, 'december': 12}
_month_like_skipped = []
for c in cat_cols:
    u = full[c].nunique()
    _vals = set(full[c].astype(str).str.strip().str.lower().unique())
    _is_month = bool(_vals) and _vals <= set(_MONTH_MAP)
    # 已经贴了片段⑤/自动生成代码（它生成了 xxx_mon 或 xxx_month）-> 不重复做
    _already = any(x in full.columns for x in (c + '_mon', c + '_month'))
    if u <= 2:                                       # 二值 -> 0/1
        full[c + '_bin'] = full[c].map({v: i for i, v in enumerate(sorted(full[c].unique()))}).astype(float)
        keep_feats.append(c + '_bin')
    elif _is_month and _already:
        P('  月份列 %s 已由片段⑤/自动生成代码编码，模板跳过（避免重复）' % c)
    elif _is_month:
        # 没贴片段⑤时兜底：模板自己补一个，绝不让它掉进 LabelEncoder
        _month_like_skipped.append(c)
        full[c + '_mon'] = full[c].astype(str).str.strip().str.lower().map(_MONTH_MAP).astype(float)
        keep_feats.append(c + '_mon')
        P('  ⚠ 检测到月份列 %s 但片段⑤没贴 -> 模板自动补了 %s_mon'
          '（它含 Jan~Dec，若按字母序编码会破坏时间顺序）' % (c, c))
    elif u <= 200:                                   # 中低基数 -> LabelEncoder
        full[c + '_le'] = LabelEncoder().fit_transform(full[c]).astype(float)
        keep_feats.append(c + '_le')
    else:                                            # 高基数 -> 靠片段③/④
        P('  ⚠ %s 基数 %d 过大，跳过编码（建议贴片段④频次编码）' % (c, u))
if _month_like_skipped:
    P('  ↑ 建议还是贴上片段⑤，让生成代码里也带上月份编码（更清楚）')

feat = [c for c in (num_cols + new_feats + keep_feats)
        if c in full.columns and full[c].notna().any()]
_seen = set()                                        # 去重（别写成 set() 推导式，会变空列表）
feat = [c for c in feat if not (c in _seen or _seen.add(c))]
nun = full.loc[is_tr, feat].nunique()
feat = [c for c in feat if nun[c] > 1]               # 去常量列

# 统一转 float32：大表下内存和训练速度都明显更好
tr_df = full[is_tr].reset_index(drop=True)
te_df = full[~is_tr].reset_index(drop=True)

# 练习超大表专用：ROW_CAP>0 时把训练集分层抽样（测试集不动，提交行数不受影响）
if ROW_CAP and len(tr_df) > ROW_CAP:
    _b = tr_df[TARGET].astype(int)
    _idx = (tr_df.assign(__b=_b).groupby('__b', group_keys=False)
              .apply(lambda d: d.sample(n=max(1, int(ROW_CAP * len(d) / len(tr_df))),
                                        random_state=SEED)).index)
    P('ROW_CAP=%d -> 训练集从 %d 行分层抽样到 %d 行（只影响训练速度与 OOF 指标口径）'
      % (ROW_CAP, len(tr_df), len(_idx)))
    tr_df = tr_df.loc[_idx].reset_index(drop=True)

X = tr_df[feat].astype(np.float32)
y = tr_df[TARGET].astype(int).values
X_test = te_df[feat].astype(np.float32)
test_ids = te_df[ID_COL]

assert list(X.columns) == list(X_test.columns), '训练/测试列数不一致！'
bad = int(X.isna().sum().sum() + X_test.isna().sum().sum())
assert bad == 0, '还有 %d 个空值，回第 3 节补填充' % bad
P('\n★ 特征 %d 个（原始数值 %d + 片段新增 %d + 编码 %d）| X %s | X_test %s'
  % (len(feat), len(num_cols), len(new_feats), len(keep_feats), X.shape, X_test.shape))
if not new_feats and not keep_feats:
    P('  ⚠ 还没有粘贴任何特征片段 —— F1 大概率停在"基础预处理"档')
tick('特征矩阵就绪')


# ============================================================================
# 6. 工具函数（阈值精确搜索用的是 O(n log n) 扫描算法，不是逐门坎重算）
# ============================================================================
def best_threshold(y_true, proba, grid=None):
    """F1 最优门坎。

    算法：按概率从大到小排序，逐个把样本纳入正类，用累积 TP/FP 一次算出所有
    切点的 F1 -> O(n log n)。旧写法（对每个门坎都调一次 f1_score）是 O(n²)，
    28 万行时等于上亿次运算，看起来就像卡死。

    ⚠ 并列概率的坑（实测踩过，F1 直接算错 0.24）：
      如果用"排名第 k 位"选门坎、却用 `(proba >= t)` 去复核，两者不是同一个
      混淆矩阵 —— 因为有并列值时，阈值会把整组并列样本一起带进来。
      所以这里【只在并列组的最后一个位置】取候选 k，这样每个候选都对应一个
      能被 `(proba >= t)` 精确复现的阈值；返回的 F1 也直接用扫描结果，不再复算。
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    order = np.argsort(-p, kind='mergesort')      # 稳定排序，保证并列组内顺序确定
    ys, ps = y_true[order], p[order]
    tp = np.cumsum(ys)                            # 前 k 个都判正类时的 TP
    k = np.arange(1, len(ys) + 1)
    fp = k - tp
    P_ = int(tp[-1]) if len(tp) else 0
    denom = 2 * tp + fp + (P_ - tp)
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(len(ys)), where=denom > 0)
    # 候选 k = 每个"并列概率组"的最后一个位置（该处的阈值可被 >= 精确复现）
    last_of_group = np.where(np.r_[ps[:-1] != ps[1:], True])[0]
    if grid is not None:                          # 只在给定门坎集合里挑
        g = np.sort(np.asarray(grid, dtype=float))
        cand, kk = [], []
        for t in g:
            pos = int(np.searchsorted(-ps, -t, side='left'))   # 概率 >= t 的样本数
            if pos > 0:
                cand.append(t)
                kk.append(pos)
        if not cand:
            return 0.5, 0.0
        kk = np.asarray(kk)
        vals = np.where(kk > 0, f1[np.clip(kk - 1, 0, len(f1) - 1)], 0.0)
        j = int(np.argmax(vals))
        return float(cand[j]), float(vals[j])
    j = int(last_of_group[int(np.argmax(f1[last_of_group]))])
    return float(ps[j]), float(f1[j])


def f1_at(y_true, proba, thr):
    return f1_score(y_true, (np.asarray(proba) >= thr).astype(int))


def check_threshold_consistency(y_true, proba, thr, f1_sweep):
    """自检：扫描法给的 F1 与用该门坎实际预测算出来的 F1 必须一致。

    不一致就说明门坎落在并列概率组中间了 —— 实测会让 F1 从 0.2388 变成 0.0000。
    返回 (是否一致, f1_score 复核值, 实际预测正类数, 扫描法对应的正类数)。
    """
    n_pred = int((np.asarray(proba) >= thr).sum())
    f1_real = f1_at(y_true, proba, thr)
    ok = abs(f1_real - f1_sweep) < 1e-6
    return ok, f1_real, n_pred, int(np.sum(np.asarray(proba) >= thr))


# ============================================================================
# 7. 建模：多模型 K 折 OOF + 集成
# ============================================================================
N_FOLDS = 3 if QUICK else 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
ES, MAX_EST = 100, 600
spw = (y == 0).sum() / max((y == 1).sum(), 1)

# RF 树数按数据量缩放：实测 6 万行 300 棵树要 213 秒，而 LGBM 只要 2 秒。
# sklearn 的 RF 建的是完全生长的树、不早停、Python 层开销大，只适合小数据。
# 它的作用只是"和 boosting 互补"，不需要很多棵树 —— 100 棵足够。
RF_TREES = 300 if len(y) <= 30000 else 100
LEAN = len(y) > 150000          # 超大表：RF 再减半（练习大表时用）

MODELS = {
    'LGBM': LGBMClassifier(learning_rate=0.02, num_leaves=127, min_child_samples=20,
                           subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=spw,
                           random_state=SEED, n_jobs=N_JOBS, verbose=-1),
    'XGB':  XGBClassifier(learning_rate=0.02, max_depth=7, min_child_weight=5,
                          subsample=0.9, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                          scale_pos_weight=spw, random_state=SEED, n_jobs=N_JOBS, verbosity=0),
    'RF':   RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=3,
                                   class_weight='balanced_subsample',
                                   random_state=SEED, n_jobs=N_JOBS),
}
if LEAN:                        # 超大表：RF 再降到 50 棵，避免它一个人吃掉几分钟
    MODELS['RF'].set_params(n_estimators=50)
P('RF 树数 = %d（按训练集 %d 行缩放）' % (MODELS['RF'].n_estimators, len(y)))
if USE_CATBOOST and _HAS_CATBOOST:
    # CatBoost 是唯一值得考虑 GPU 的模型，但要按数据量决定（见文件头的实测表）：
    #   2.4 万行 -> CPU 快 4.1 倍；12 万行打平；28 万行 GPU 快 3.4 倍
    if str(USE_GPU).lower() == 'auto':
        _use_cb_gpu = len(y) > GPU_MIN_ROWS
        P('GPU 自动判断：训练 %d 行 %s 门槛 %d -> CatBoost 用 %s'
          % (len(y), '>' if len(y) > GPU_MIN_ROWS else '<=', GPU_MIN_ROWS,
             'GPU' if _use_cb_gpu else 'CPU'))
    else:
        _use_cb_gpu = bool(USE_GPU)
        P('GPU 手动指定：CatBoost 用 %s（注意小数据上 GPU 可能是负优化）'
          % ('GPU' if _use_cb_gpu else 'CPU'))
    _cb_kw = dict(task_type='GPU', devices='0') if _use_cb_gpu else {}
    MODELS['CatBoost'] = CatBoostClassifier(
        iterations=MAX_EST, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        loss_function='Logloss', eval_metric='Logloss', od_type='Iter', od_wait=ES,
        auto_class_weights='Balanced' if spw > 1.2 else None,
        random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=N_JOBS, **_cb_kw)
if QUICK:
    MODELS = {k: v for k, v in MODELS.items() if k in ('LGBM', 'RF')}


def best_iter_from_curve(model, metric='binary_logloss', min_iter=100, smooth=15):
    """从 evals_result_ 里手动找最优轮数（不依赖早停回调）。

    为什么手动：实测 lightgbm 4.6 + pandas 数据时 early_stopping 回调会失效 ——
    best_iteration_ 报 2，实际却跑满 2000 轮（既慢又欠拟合）。
    做法：对验证曲线做滑动平均（窗口 smooth）再取最小值，并设最小轮数下限，
    避免"因为小验证集噪声而在第 20 轮就停"。
    """
    try:
        ev = model.evals_result_['valid_0']
        key = metric if metric in ev else list(ev.keys())[0]
        curve = np.asarray(ev[key], dtype=float)
        if len(curve) >= smooth * 2:
            ker = np.ones(smooth) / smooth
            curve = np.convolve(curve, ker, mode='valid')       # 平滑后找最小点
            best = int(np.argmin(curve)) + smooth                # 补回窗口偏移
        else:
            best = int(np.argmin(curve)) + 1
        return max(min_iter, best)
    except Exception:
        return None


def fit_es(m, Xa, ya, Xb, yb, n_est=MAX_EST):
    """带早停的 fit。四个注意点（都是实测踩出来的）：
      ① xgboost 2.x 的 early_stopping_rounds 只能在 fit()，3.x 只能在构造器
      ② catboost 用 od_type/od_wait 且参数名是 iterations（同时传 n_estimators 会报错）；
         早停指标一律用 logloss（AUC 在小验证集上是噪声）
      ③ lightgbm 的回调会正常提前停止，但它报的 best_iteration_ 是错的
         （实测：报 3，实际停在 103 轮）-> 轮数一律自己从平滑曲线算
      ④ 曲线最优值必须做平滑 + 最小轮数下限，否则会在第 20 轮就停（欠拟合）
    """
    if isinstance(m, LGBMClassifier):
        m.set_params(n_estimators=n_est)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='binary_logloss',
              callbacks=[lgb_es(ES, verbose=False)])
        curve_len = len(np.asarray(m.evals_result_['valid_0']['binary_logloss']))
        n_cb = getattr(m, 'best_iteration_', None)
        if DEBUG_ES:
            P('      [ES探针] 设 n_est=%d | 回调报 best_iteration_=%s | 实际训练轮数=%d'
              % (n_est, n_cb, curve_len))
        # 实测（lightgbm 4.6 + pandas）：回调会正常提前停止（103 轮就停），
        # 但它报的 best_iteration_ 是错的（报 3）。所以只信"提前停止"这个行为
        # （用它省时间），轮数一律用平滑曲线自己算。
        return best_iter_from_curve(m) or n_est
    if isinstance(m, XGBClassifier):
        if _XGB_MAJOR >= 3:
            m.set_params(n_estimators=n_est, early_stopping_rounds=ES, eval_metric='logloss')
        else:
            m.set_params(n_estimators=n_est, early_stopping_rounds=ES)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        n_x = getattr(m, 'best_iteration', None)
        if DEBUG_ES:
            P('      [ES探针] XGB best_iteration=%s' % n_x)
        return n_x or n_est
    if _HAS_CATBOOST and isinstance(m, CatBoostClassifier):
        m.set_params(eval_metric='Logloss')
        m.fit(Xa, ya, eval_set=(Xb, yb), verbose=False)
        return m.get_best_iteration() or n_est
    # 其它模型（RF/ExtraTrees…）没有早停：直接 fit，并返回 None。
    # ⚠ 必须返回 None，不能返回 n_est！否则外层会把"复用的轮数"当成
    #   n_estimators 塞回去，把 RF 从 100 棵变成 2000 棵（实测慢 20 倍：4s -> 90s）。
    m.fit(Xa, ya)
    return None


def oof_proba(model, tag=''):
    """K 折 OOF 概率 + 测试集多折平均。

    两处提速（大表下很关键）：
      · 早停验证集只取 ES_FRAC（6%），只用来判断轮数
      · REUSE_BEST_ITER=True 时，第 1 折早停得出的轮数给其余折复用
    每折打印耗时（tag 非空时），卡住时能立刻看出是哪一折哪一步慢。
    """
    oof, tp = np.zeros(len(y)), np.zeros(len(X_test))
    best_iter = None
    for i, (tr, va) in enumerate(skf.split(X, y)):
        m = model.__class__(**model.get_params())
        t_fit = time.time()
        if best_iter and REUSE_BEST_ITER:
            # 复用第 1 折得出的轮数：这一折不再切早停集，全部训练数据都用上
            # ⚠ CatBoost 不接受同时传 iterations 和 n_estimators（会直接报错），
            #   所以这里按模型类型选参数名。
            if _HAS_CATBOOST and isinstance(m, CatBoostClassifier):
                m.set_params(iterations=best_iter)
            else:
                m.set_params(n_estimators=best_iter)
            m.fit(X.iloc[tr], y[tr])
        else:
            Xa, Xb, ya, yb = train_test_split(X.iloc[tr], y[tr], test_size=ES_FRAC,
                                              stratify=y[tr], random_state=SEED)
            got = fit_es(m, Xa, ya, Xb, yb)
            if best_iter is None and got:
                best_iter = int(got)
        t_fit = time.time() - t_fit
        t_pred = time.time()
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(X_test)[:, 1] / N_FOLDS
        t_pred = time.time() - t_pred
        if tag:
            P('    %s 第%d折: 训练 %.1fs / 预测 %.1fs%s'
              % (tag, i + 1, t_fit, t_pred, '（复用 %d 轮）' % best_iter if best_iter else ''))
    return oof, tp


P('\n' + '=' * 78)
P('%d 折交叉验证（%d 行 × %d 特征，正类 %r）' % (N_FOLDS, len(y), len(feat), SUB_POS))
P('=' * 78)
oof_store, test_store, rows = {}, {}, []
for name, m in MODELS.items():
    t1 = time.time()
    try:
        oof, tp = oof_proba(m, tag=name)
        thr, f = best_threshold(y, oof)
        pred = (oof >= thr).astype(int)
        oof_store[name], test_store[name] = oof, tp
        rows.append(dict(模型=name, F1=round(f, 4), 门坎=round(thr, 3),
                         AUC=round(roc_auc_score(y, oof), 4),
                         P=round(precision_score(y, pred), 3),
                         R=round(recall_score(y, pred), 3),
                         秒=round(time.time() - t1, 1)))
        P('%-9s F1=%.4f  AUC=%.4f  门坎=%.3f  (%.0fs, %d折)'
          % (name, f, rows[-1]['AUC'], thr, time.time() - t1, N_FOLDS))
    except Exception as e:
        P('%-9s 失败 %s: %s' % (name, type(e).__name__, str(e)[:70]))

cv = pd.DataFrame(rows).sort_values('F1', ascending=False)
P('\n' + cv.to_string(index=False))
best_name = cv.iloc[0]['模型']
final_oof, final_test = oof_store[best_name], test_store[best_name]
final_tag = best_name
final_thr, final_f1 = best_threshold(y, final_oof)
P('★ 最佳单模型：%s  OOF F1 = %.4f' % (best_name, final_f1))

# 集成：等权软投票 / 排序平均（OOF 上更好才采用）
use = cv[cv['F1'] >= cv['F1'].max() - 0.02]['模型'].tolist()
if len(use) >= 2:
    o = np.mean([oof_store[n] for n in use], axis=0)
    t = np.mean([test_store[n] for n in use], axis=0)
    thr, f = best_threshold(y, o)
    P('软投票(%s)  OOF F1 = %.4f%s' % ('+'.join(use), f, '  ★采用' if f > final_f1 else ''))
    if f > final_f1:
        final_oof, final_test, final_thr, final_f1 = o, t, thr, f
        final_tag = '软投票(%s)' % '+'.join(use)
    from scipy.stats import rankdata
    o = np.mean([rankdata(oof_store[n]) / len(y) for n in use], axis=0)
    t = np.mean([rankdata(test_store[n]) / len(X_test) for n in use], axis=0)
    thr, f = best_threshold(y, o)
    P('排序平均(%s)  OOF F1 = %.4f%s' % ('+'.join(use), f, '  ★采用' if f > final_f1 else ''))
    if f > final_f1:
        final_oof, final_test, final_thr, final_f1 = o, t, thr, f
        final_tag = '排序平均(%s)' % '+'.join(use)
P('★ 最终方案：%s | OOF F1 = %.4f' % (final_tag, final_f1))


# ============================================================================
# 8. 门坎值优化（PPT 第 8 条；通常能白捡 0.005~0.02）
# ============================================================================
# ① 精确最优  ② 同档取中位（防过拟合 OOF 噪声）  ③ 正类率对齐（更稳的先验）
thr_exact, f1_exact = best_threshold(y, final_oof)
grid = np.arange(0.005, 0.996, 0.005)
f1s = np.array([f1_at(y, final_oof, t) for t in grid])
plateau = grid[f1s >= f1_exact - 0.002]
safe_thr = float(np.clip(0.5 * thr_exact + 0.5 * float(np.median(plateau)), 0.005, 0.995))
rate_thr = float(np.quantile(final_oof, 1 - y.mean()))
f1_rate = f1_at(y, final_oof, rate_thr)
P('默认 0.5    -> F1 = %.4f' % f1_at(y, final_oof, 0.5))
P('精确最优 %.3f -> F1 = %.4f' % (thr_exact, f1_exact))
P('稳健化   %.3f -> F1 = %.4f' % (safe_thr, f1_at(y, final_oof, safe_thr)))
P('正类率对齐 %.3f -> F1 = %.4f' % (rate_thr, f1_rate))
# ★ 诊断：三条候选各自会让【训练集】多少比例被判为正类。
#   如果某条候选把这个比例推到训练集正类率的 2 倍以上，就要小心 ——
#   它在 test 上可能更激进（实测 2606 电商：正类率对齐那条候选让 test 的
#   正类占比从 13.6% 变成 43%，虽然 OOF F1 显示它"更好"，但那是训练集上的口径）。
P('  候选门坎对应的训练集正类占比: 精确最优 %.3f | 稳健化 %.3f | 正类率对齐 %.3f（训练集正类率 %.3f）'
  % ((final_oof >= thr_exact).mean(), (final_oof >= safe_thr).mean(),
     (final_oof >= rate_thr).mean(), y.mean()))
final_thr = rate_thr if f1_rate > f1_at(y, final_oof, safe_thr) else safe_thr
pred = (final_oof >= final_thr).astype(int)
final_f1 = f1_score(y, pred)
P('★ 采用门坎 %.3f -> OOF F1 = %.4f  (P=%.3f R=%.3f)'
  % (final_thr, final_f1, precision_score(y, pred), recall_score(y, pred)))

# 参考线：全猜正类的 F1 = 2p/(1+p)。低于它说明模型没学到东西（p=0.35 时约 0.519）
floor_f1 = 2 * float(y.mean()) / (1 + float(y.mean()))
P('参考：全猜正类 F1 = %.4f（正类率 %.3f）| 你比它高 %+.4f'
  % (floor_f1, y.mean(), final_f1 - floor_f1))


# ============================================================================
# 9. 写提交 + 格式自检（格式错 = 0 分，这一段一定跑）
# ============================================================================
def write_submission(proba, thr, path=SUB_PATH):
    pred01 = (np.asarray(proba) >= thr).astype(int)
    lab = [SUB_POS if p == 1 else SUB_NEG for p in pred01]
    out = pd.DataFrame({ID_COL: np.asarray(test_ids), SUB_COL: lab})
    out.to_csv(path, index=False, encoding='utf-8-sig')
    P('已写出 %s  形状=%s' % (path, out.shape))
    P(out[SUB_COL].value_counts().to_string())
    return out


sub = write_submission(final_test, final_thr, SUB_PATH)
_raw_check = pd.read_csv(SUB_PATH, encoding='utf-8-sig', dtype=str)

# ★★ 兜底：如果预测【全部是同一类】，提交就是 0 分。
#    实测踩过（特征极弱的老数据 + ID 列被当特征）：模型概率全被压到门坎以下，
#    最后一次 assert 直接把整个脚本打挂 —— 考场上这是最坏的结果
#    （明明还有时间，却因为一行 assert 拿不到任何提交）。
#    这里改成"自动补救"：忽略绝对门坎，改成【按概率取前 K% 判为正类】，
#    K 取训练集正类率（题目说约 35%，这也是最稳的估计）。
if _raw_check[SUB_COL].nunique() < 2:
    _k = float(np.clip(y.mean(), 0.02, 0.98))
    _cut = np.quantile(final_test, 1 - _k)
    P('\n⚠ 检测到预测全部是同一类（这会是 0 分）-> 自动补救：'
      '改成按概率取前 %.1f%% 判为正类（= 训练集正类率）' % (_k * 100))
    P('   补救前门坎 %.4f 让 %d/%d 行判为正类；补救后门坎 %.4f'
      % (final_thr, int((np.asarray(final_test) >= final_thr).sum()),
         len(final_test), _cut))
    sub = write_submission(final_test, _cut, SUB_PATH)

# 必须 dtype=str 读回来：否则 pandas 会把 TRUE/FALSE 推断成布尔，自检会误报"取值不对"
raw = pd.read_csv(SUB_PATH, encoding='utf-8-sig', dtype=str)
vals = sorted(raw[SUB_COL].unique())
P('\n===== 提交自检 =====')
P('列名 %s | 行数 %d（测试集 %d）' % (list(raw.columns), len(raw), len(X_test)))
P('取值 %s（必须是 [%s, %s] 这两个，大小写照 docx）' % (vals, SUB_NEG, SUB_POS))
assert len(raw.columns) == 2 and raw.columns[0] == ID_COL, '列名不对！'
assert len(raw) == len(X_test), '行数对不上！'
assert raw[SUB_COL].isna().sum() == 0, '有缺失值！'
assert set(vals) <= {SUB_POS, SUB_NEG}, '取值不对（大小写要照 docx）！'
assert raw[SUB_COL].nunique() >= 2, '全部预测同一个值 = 0 分！'
P('正类占比 %.4f（训练集 %.4f）' % ((raw[SUB_COL] == SUB_POS).mean(), y.mean()))
P('✓ 格式检查通过')

P('\n===== 汇总 =====')
P('方案 %s | 特征 %d 个 | 正类 %s | OOF F1 %.4f | 档位: %s'
  % (final_tag, len(feat), SUB_POS, final_f1,
     '>=0.60 优秀' if final_f1 >= 0.60 else
     '0.56~0.59 强特征工程+调参' if final_f1 >= 0.56 else
     '0.50~0.55 特征工程有效' if final_f1 >= 0.50 else
     '0.40~0.49 基础预处理' if final_f1 >= 0.40 else
     '<0.40 需要加强预处理/特征工程'))
P('总耗时 %.0f 秒（线程 %d）| 交卷：CSV + 代码 一起放进「姓名+证件号」文件夹'
  % (time.time() - T0, N_JOBS))


# ============================================================================
# 10. 加分实验（QUICK=False 才跑；只有更好才采用，都有超时保护）
# ============================================================================
if QUICK:
    P('\n[加分实验] QUICK=True -> 跳过')
else:
    P('\n' + '=' * 78)
    P('加分实验（每项都比 OOF F1，只有更好才采用）')
    P('=' * 78)
    base_f1 = final_f1

    # A. 不平衡策略：不加权 + 门坎优化（有时比加权更好）
    try:
        t1 = time.time()
        p0 = dict(MODELS['LGBM'].get_params()); p0['scale_pos_weight'] = 1.0
        o0, t0_ = oof_proba(LGBMClassifier(**p0))
        thr0, f0 = best_threshold(y, o0)
        P('A. LGBM 不加权    F1 = %.4f (%.0fs) | 当前 %.4f%s'
          % (f0, time.time() - t1, final_f1, '  ★采用' if f0 > final_f1 else ''))
        if f0 > final_f1:
            final_oof, final_test, final_thr, final_f1, final_tag = o0, t0_, thr0, f0, 'LGBM(不加权)'
    except Exception as e:
        P('A. 失败:', type(e).__name__, str(e)[:60])

    # B. LGBM 轻量调参：坐标轮换一轮，只调最关键的 3 个参数（120 秒安全阀）
    try:
        t1 = time.time()
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        Xa1, Xa2, ya1, ya2 = train_test_split(Xa, ya, test_size=ES_FRAC, stratify=ya,
                                              random_state=SEED)

        def score(p):
            mm = LGBMClassifier(**p)
            mm.set_params(n_estimators=MAX_EST)
            mm.fit(Xa1, ya1, eval_set=[(Xa2, ya2)], eval_metric='auc',
                   callbacks=[lgb_es(ES, verbose=False)])
            return best_threshold(yb, mm.predict_proba(Xb)[:, 1])[1]

        best_p = dict(MODELS['LGBM'].get_params())
        cur = score(best_p)
        for key, cands in [('num_leaves', [31, 63, 255]),
                           ('min_child_samples', [5, 50, 100]),
                           ('colsample_bytree', [0.6, 0.9])]:
            if time.time() - t1 > 120:
                P('   （调参时间预算到，停止搜索）')
                break
            for v in cands:
                trial = dict(best_p); trial[key] = v
                s = score(trial)
                if s > cur + 1e-4:
                    cur, best_p = s, trial
                    P('   %s = %s -> %.4f ★' % (key, v, s))
        o1, t1_ = oof_proba(LGBMClassifier(**best_p))
        thr1, f1 = best_threshold(y, o1)
        P('B. LGBM 调参后    F1 = %.4f (%.0fs) | 当前 %.4f%s'
          % (f1, time.time() - t1, final_f1, '  ★采用' if f1 > final_f1 else ''))
        if f1 > final_f1:
            final_oof, final_test, final_thr, final_f1, final_tag = o1, t1_, thr1, f1, 'LGBM(调参)'
    except Exception as e:
        P('B. 失败:', type(e).__name__, str(e)[:60])

    # C. 特征工程到底有没有用（PPT 就按这个分档，出个证据）
    #   ★关键：基线必须包含"类别编码"，否则对比不公平 ——
    #     曾经的写法是 only_raw = 只有原始数值列 vs 全部特征，
    #     但"全部特征"里还含类别编码列，于是"特征工程贡献"里混进了编码的功劳，
    #     数字会虚高。这里改成：基线 = 原始数值 + 编码，实验 = 再加衍生特征，
    #     两者只差"衍生特征"，差值才是真正的特征工程贡献。
    try:
        enc_like = [c for c in feat
                    if c in num_cols or c.endswith(('_bin', '_mon', '_le', '_ord', '_te'))]
        if enc_like and len(enc_like) < len(feat):
            Xa, Xb, ya, yb = train_test_split(tr_df[enc_like].astype(np.float32), y,
                                              test_size=0.2, stratify=y, random_state=SEED)
            m2 = LGBMClassifier(**MODELS['LGBM'].get_params())
            m2.fit(Xa, ya)
            f_base = best_threshold(yb, m2.predict_proba(Xb)[:, 1])[1]
            # 实验组：同样的切分，加上衍生特征
            Xa2, Xb2, _, _ = train_test_split(tr_df[feat].astype(np.float32), y,
                                              test_size=0.2, stratify=y, random_state=SEED)
            m3 = LGBMClassifier(**MODELS['LGBM'].get_params())
            m3.fit(Xa2, ya)
            f_full = best_threshold(yb, m3.predict_proba(Xb2)[:, 1])[1]
            P('C. 特征工程对比（同一份切分，只差衍生特征）:')
            P('   基线 = 原始数值+类别编码 %d 列      -> F1 = %.4f' % (len(enc_like), f_base))
            P('   实验 = 再加衍生特征 %d 列          -> F1 = %.4f' % (len(feat), f_full))
            P('   => 衍生特征的真实贡献 %+.4f%s'
              % (f_full - f_base,
                 '  （有正贡献，可保留）' if f_full > f_base + 0.0005 else
                 '  （★没提升甚至下降，建议把片段删掉，用干净数据）'))
    except Exception as e:
        P('C. 失败:', type(e).__name__, str(e)[:60])

    if final_f1 > base_f1:                            # 有提升就重写提交并复核
        P('\n★ 加分实验有提升（%.4f -> %.4f），重写提交文件' % (base_f1, final_f1))
        write_submission(final_test, final_thr, SUB_PATH)
        raw = pd.read_csv(SUB_PATH, encoding='utf-8-sig', dtype=str)
        assert set(raw[SUB_COL].unique()) <= {SUB_POS, SUB_NEG}, '取值不对！'
        assert raw[SUB_COL].nunique() >= 2, '全部预测同一个值 = 0 分！'
        P('✓ 最终提交复核通过：方案 %s | OOF F1 = %.4f' % (final_tag, final_f1))
    else:
        P('\n加分实验没有超过当前方案，保持原结果（%.4f）' % final_f1)

# ============================================================================
# 大表还嫌慢时的排查顺序（按效果排序）：
#   1) N_JOBS 调到机器物理核数的一半（比如 8 核用 4）；确认没有别的任务在抢 CPU
#   2) QUICK = True（只跑 LGBM+RF、3 折、跳过分实验）——最快能出结果
#   3) USE_CATBOOST = False（CatBoost 比 LGBM 慢一倍以上）
#   4) 片段① 的 PAIR_TOP_N 调小（特征越多越慢），或减少粘贴的片段数量
#   5) 确认没在 28 万行这种超大表上硬跑：真实考题只有 2.4 万行，正常 3~5 分钟
# ============================================================================
