# ============================================================================
# 模板⑤：文本对相似度（判断两句话是否语义相似）
# 适用：202309 语义相似性(Sentence1 vs Sentence2 -> Similar 0/1)
# 评分：预测为 1 的 F1
#
# 【本数据集实测 F1（5折CV × 5个随机种子，标准差均 ≈0.0008）】
#   全预测 1 的平凡基线 .................... 0.8063   <-- 打不过它就等于白做
#   旧版模板(TF-IDF cos+dot) ............... 0.8026   <-- 比平凡基线还低
#   本版 离线档(纯 TF-IDF, 17维) ........... 0.8295
#   本版 满血档(+ bge-m3 语义特征, 50维) .... 0.8446
#
# 【两档设计】USE_EMBEDDING=True 时自动找本地模型；找不到就自动退回离线档，
#            不会报错中断。考场没网/模型没带，代码照样跑完出结果。
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import os
import re
import difflib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ===== 改这里 1：路径 + 两个文本列 + 目标列 + ID列 =====
train = pd.read_excel('Training.xlsx', index_col='Pair_Num')
test  = pd.read_excel('Test.xlsx',     index_col='Pair_Num')
COL1, COL2 = 'Sentence1', 'Sentence2'
TARGET = 'Similar'
ID_COL = 'Pair_Num'
OUT_CSV = '考生姓名_results.csv'

USE_EMBEDDING = True          # 有本地模型就用；没有会自动退回纯 TF-IDF
EMB_MODEL = 'bge-m3'          # models/ 下的文件夹名。中英文都强，实测比 Qwen3 好
# ====================================================


# ---------- 1. 补缺失 + 清洗 ----------
# 两列都要补：本数据 Sentence2 有 20 个缺失，只补一列会漏
for d in (train, test):
    d[COL1] = d[COL1].fillna('')
    d[COL2] = d[COL2].fillna('')

def clean(t):
    """小写化 + 只保留 字母/汉字/数字/空格。给 TF-IDF 用。"""
    if not isinstance(t, str):
        return ''
    return re.sub(r'[^a-zA-Z一-鿿0-9 ]', ' ', t.lower())

for d in (train, test):
    d['c1'] = d[COL1].apply(clean)
    d['c2'] = d[COL2].apply(clean)

y = train[TARGET].astype(int).values


# ---------- 2. TF-IDF：train+test 全体文本一起 fit ----------
all_text = pd.concat([train['c1'], train['c2'], test['c1'], test['c2']]).tolist()
WORD = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2).fit(all_text)
CHAR = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2,
                       sublinear_tf=True).fit(all_text)

def cos_rows(a, b):
    """逐行余弦（向量化）。

    TfidfVectorizer 默认 norm='l2'，向量已单位化 -> 余弦 == 点积。
    比 for 循环调 cosine_similarity 快 15 倍。
    （旧版模板把 cos 和 dot 当两个特征用，实测相关系数 1.000000，纯属重复。）
    """
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


# ---------- 3. 特征工程：多视角相似度 ----------
NUM = re.compile(r'\d+\.?\d*')
STOP = set(('a an the of in on at to for and or is are was were be been it its '
            'this that with as by from has have had').split())

def build_feats(df):
    """单个相似度顶不住，用一组互补视角：词/字符/集合/序列/长度/数字。"""
    s1, s2 = df['c1'].tolist(), df['c2'].tolist()
    S1 = [set(a.split()) for a in s1]
    S2 = [set(b.split()) for b in s2]
    f = {}

    # 3.1 向量空间相似度
    f['cos_word'] = cos_rows(WORD.transform(df['c1']), WORD.transform(df['c2']))
    f['cos_char'] = cos_rows(CHAR.transform(df['c1']), CHAR.transform(df['c2']))

    # 3.2 集合重叠
    f['jaccard'] = np.array([len(a & b) / max(len(a | b), 1) for a, b in zip(S1, S2)])
    f['contain'] = np.array([len(a & b) / max(min(len(a), len(b)), 1) for a, b in zip(S1, S2)])
    f['prec']    = np.array([len(a & b) / max(len(a), 1) for a, b in zip(S1, S2)])
    f['rec']     = np.array([len(a & b) / max(len(b), 1) for a, b in zip(S1, S2)])
    N1 = [a - STOP for a in S1]
    N2 = [b - STOP for b in S2]
    f['jac_nostop'] = np.array([len(a & b) / max(len(a | b), 1) for a, b in zip(N1, N2)])

    # 3.3 词序 / 序列
    def bigrams(t):
        w = t.split()
        return set(zip(w, w[1:]))
    B1 = [bigrams(a) for a in s1]
    B2 = [bigrams(b) for b in s2]
    f['jac_bigram'] = np.array([len(a & b) / max(len(a | b), 1) if (a or b) else 0.
                                for a, b in zip(B1, B2)])
    f['seqratio'] = np.array([difflib.SequenceMatcher(None, a, b).ratio()
                              for a, b in zip(s1, s2)])
    f['lcs'] = np.array([difflib.SequenceMatcher(None, a.split(), b.split()).ratio()
                         for a, b in zip(s1, s2)])

    # 3.4 长度
    L1 = np.array([len(a.split()) for a in s1], float)
    L2 = np.array([len(b.split()) for b in s2], float)
    f['len_diff']  = np.abs(L1 - L2)
    f['len_ratio'] = np.minimum(L1, L2) / np.maximum(np.maximum(L1, L2), 1)
    f['len_mean']  = (L1 + L2) / 2
    f['only1'] = np.array([len(a - b) for a, b in zip(S1, S2)], float)
    f['only2'] = np.array([len(b - a) for a, b in zip(S1, S2)], float)

    # 3.5 数字一致性（从【原文】提取，clean 会破坏 $ % . 等符号）
    n1 = [set(NUM.findall(str(t))) for t in df[COL1]]
    n2 = [set(NUM.findall(str(t))) for t in df[COL2]]
    f['num_jac']  = np.array([len(a & b) / max(len(a | b), 1) if (a or b) else 1.
                              for a, b in zip(n1, n2)])
    f['num_diff'] = np.array([len(a ^ b) for a, b in zip(n1, n2)], float)

    return pd.DataFrame(f)

Ftr = build_feats(train)
Fte = build_feats(test)
X, X_test = Ftr.values, Fte.values
mode = '离线档 (纯 TF-IDF)'


# ---------- 4.（可选）语义 embedding 特征 ----------
def find_model(name):
    """在当前目录及上级目录里找 models/<name>。找不到返回 None。"""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        p = os.path.join(d, 'models', name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'config.json')):
            return p
        d = os.path.dirname(d)
    return None

if USE_EMBEDDING:
    mp = find_model(EMB_MODEL)
    if mp is None:
        print(f'[提示] 没找到本地模型 models/{EMB_MODEL}，自动使用离线档。')
    else:
        try:
            # 必须在 import sentence_transformers 之前设置：
            os.environ.pop('HF_ENDPOINT', None)      # 别用 hf-mirror，会让 hub 校验失败
            os.environ['HF_HUB_OFFLINE'] = '1'       # 纯本地，避免联网卡顿
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            import ssl                               # Win 证书库有坏证书，py3.9 会抛 SSLError
            _orig = ssl.SSLContext.load_default_certs
            def _safe(self, *a, **k):
                try: _orig(self, *a, **k)
                except ssl.SSLError: pass
            ssl.SSLContext.load_default_certs = _safe

            from sentence_transformers import SentenceTransformer
            import torch

            cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 f'_emb_{EMB_MODEL}.npz')
            if os.path.exists(cache):
                z = np.load(cache)
                e1, e2, t1, t2 = z['e1'], z['e2'], z['t1'], z['t2']
                print(f'[embedding] 读取缓存 {os.path.basename(cache)}')
            else:
                dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                m = SentenceTransformer(mp, device=dev)
                # 整批传 list，别用 .apply(model.encode) 逐条编码（慢几十倍）
                # 喂【原文】不喂 clean()：transformer 需要标点和大小写
                kw = dict(batch_size=64, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)
                print(f'[embedding] {EMB_MODEL} 编码中 (device={dev}) ...')
                e1 = m.encode(train[COL1].tolist(), **kw)
                e2 = m.encode(train[COL2].tolist(), **kw)
                t1 = m.encode(test[COL1].tolist(), **kw)
                t2 = m.encode(test[COL2].tolist(), **kw)
                np.savez_compressed(cache, e1=e1, e2=e2, t1=t1, t2=t2)
                del m

            # 已归一化 -> 点积即余弦，不需要 util.cos_sim 循环
            cos_tr = (e1 * e2).sum(1, keepdims=True)
            cos_te = (t1 * t2).sum(1, keepdims=True)
            # 差向量降到 32 维再用。直接上 1024 维 |a-b| 会过拟合（实测掉到 0.81）
            svd = TruncatedSVD(n_components=32, random_state=42).fit(np.abs(e1 - e2))
            X      = np.hstack([X,      cos_tr, svd.transform(np.abs(e1 - e2))])
            X_test = np.hstack([X_test, cos_te, svd.transform(np.abs(t1 - t2))])
            mode = f'满血档 (TF-IDF + {EMB_MODEL})'
        except Exception as e:
            print(f'[警告] embedding 失败，退回离线档: {type(e).__name__}: {str(e)[:120]}')

print(f'模式: {mode}   特征维度: {X.shape}   测试: {X_test.shape}')


# ---------- 5. 训练 + 阈值调优 ----------
# 关键：评分只看正类 F1，而正类占 67.5%，默认 0.5 阈值不是最优。
# 用交叉验证的 out-of-fold 概率选阈值，避免在训练集上自欺欺人。
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]

grid = np.arange(0.05, 0.96, 0.01)
scores = [f1_score(y, (oof >= t).astype(int)) for t in grid]
BEST_T = float(grid[int(np.argmax(scores))])

base_f1 = f1_score(y, np.ones_like(y))
print('-' * 62)
print(f'平凡基线(全预测1) F1 : {base_f1:.4f}')
print(f'交叉验证 F1 @0.50    : {f1_score(y, (oof >= 0.5).astype(int)):.4f}')
print(f'交叉验证 F1 @{BEST_T:.2f}    : {max(scores):.4f}   <-- 采用此阈值')
print('-' * 62)
if max(scores) <= base_f1:
    print('⚠ 模型没打过"全预测1"，建议直接全填 1 交卷，或回头查特征。')

clf.fit(X, y)
pred = (clf.predict_proba(X_test)[:, 1] >= BEST_T).astype(int)


# ---------- 6. 输出 ----------
result = pd.DataFrame({ID_COL: test.index, 'Predicted_Result': pred})
result.to_csv(OUT_CSV, index=False)
print(f'已写出 {OUT_CSV}   预测为1比例: {pred.mean():.3f}  (训练集真实比例 {y.mean():.3f})')
print(result.head())


# ============================================================================
# 环境备忘（本机实测，2026-08）
# ----------------------------------------------------------------------------
# conda 环境 pytorch: Python 3.9.23 / torch 2.8.0+cu126(RTX 4060) /
#                     transformers 4.57.6 / sentence-transformers 5.1.2
#
# Python 3.9 的天花板：transformers 最高 4.57.6、sentence-transformers 最高 5.1.2
#   （再往上的 v5 / v6 都要求 Python>=3.10）。想用更新的版本得先升 Python。
#
# 三个已踩过的坑：
#   1. 【别设 HF_ENDPOINT=hf-mirror.com】该站只做 308 跳转、不是真镜像，
#      会导致 huggingface_hub 元数据校验失败。本机有系统代理，直连 HF 正常。
#   2. Windows 证书库有损坏证书 -> py3.9 的 ssl.py 抛 SSLError
#      -> 见上面 _safe 补丁（必须在 import sentence_transformers 之前打）
#   3. .apply(model.encode) 逐条编码慢几十倍，一定要整批传 list
#
# 模型选型实测（跨 4 套历年真题，非只看本题）：
#                        本题(对称)  商品检索   阅读理解   问答T/F
#   bge-m3    (采用)      0.8446     89.68%    33.40%    77.89%
#   MiniLM    (仅英文)    0.8402       --        --        --
#   Qwen3-0.6B(已删)      0.8360     87.09%    37.70%    78.05%
#   多数类/随机基线          0.8063       --     25.00%    78.05%
#
#   结论：bge-m3 在 4 类任务里赢 3 类，且体积小一半、快 2~8 倍，是唯一需要的。
#   Qwen3 只在【非对称 query->doc】的阅读理解上领先，但 37.7% 绝对分太低，
#   那类题该用生成式方案(官方解上的 flan-t5)，靠 embedding 余弦拿不到分。
#   问答T/F 两个模型都没打过多数类基线 -> 该题型 embedding 余弦直接无效。
#
# 【重要】"新模型 = 更好" 不成立，选型要看任务是【对称相似】还是【非对称检索】。
#
# 本地模型位置: 练习/models/{bge-m3, all-MiniLM-L6-v2}
# 重下模型: python _download_model.py
# ============================================================================
