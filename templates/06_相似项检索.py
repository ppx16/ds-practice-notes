# ============================================================================
# 模板⑥：相似项检索（给 20 个测试样本，从训练集找同一产品的其它贴文，输出 ID 列表）
# 适用：202303 / 练习2303 相同产品识别（商品图片 Phash + 标题文本）
# 评分：Macro-F1 —— 每笔：precision=|预测∩真实|/|预测|，recall=|预测∩真实|/|真实|，
#       再算 F1，最后对 20 笔求平均。输出 Num 必须在 30~50 之间，否则 0 分。
# 提交格式：Posting_ID, Num, Predicted_Results（Predicted_Results 用英文逗号分隔 ID）
#
# 核心结论（实测）：
#   1. 图像 Phash 汉明距离对这道题没用（正确/错误算都是 ~32 的随机值，不同卖家
#      同一商品用不同角度的图，感知哈希聚不到一起）——官方模板①用 Phash+标题并集，
#      其实是靠标题那一路在起作用。
#   2. 标题才是主信号：char_wb(3,4) 字符 n-gram TF-IDF 余弦相似度（小写），直接
#      Top-35 就有 Macro-F1≈0.692；换成词级 ngram(1,2) 或英文 MiniLM 向量都更差。
#   3. 再叠一层「2-hop 桥接」：用 top-10 种子做一跳，把「种子→其它」相似度 ≥0.55
#      的也拉进来，与直接 Top-30 取并集，测试集 Macro-F1≈0.753（P≈0.70, R≈0.83）。
#   4. 输出条数：F1 随 Num 单调升到 50（上限），所以 Num 尽量填满 50。
# ⚠️ 离线：TF-IDF 纯本地，不联网、不下载模型。
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== 改这里 1：路径 + 列名 =====
TRAIN_PATH = 'data/train.csv'        # 训练集（检索语料库，约 3.4 万条）
TEST_PATH  = 'data/test_ans.csv'     # 测试集（20 个查询商品）
TITLE_COL  = 'Posting_Title'         # 标题文本列
ID_COL     = 'Posting_ID'            # 主键列
PHASH_COL  = 'Posting_Image_Phash'   # 图像哈希列（本模板不使用，仅 dtype 兼容）
OUT_PATH   = '练习2303_results.csv'  # 考试时改成「考生姓名_results.csv」
# ====================================================

# ===== 改这里 2：检索参数（实测最优，已验证能泛化，勿乱调）=====
DIRECT_K = 30    # 直接标题余弦相似度，取前 K 个
SEED_N   = 10    # 2-hop 桥接的种子数（用 top-N 最相似的标题当种子）
THR      = 0.55  # 桥接阈值：种子与候选的相似度 ≥ 该值才并入
MAX_NUM  = 50    # 输出上限（题目要求 30~50，填满 50 最优）
# =================================================================

# ---------- 1. 读数据 ----------
# Phash 列用 str 读，避免「纯数字十六进制被 pandas 当 float 读成 9.79E+108」的坑
train = pd.read_csv(TRAIN_PATH, dtype={PHASH_COL: str})
test  = pd.read_csv(TEST_PATH,  dtype={PHASH_COL: str})

train_title = train[TITLE_COL].fillna('').astype(str).str.lower().tolist()
test_title  = test[TITLE_COL].fillna('').astype(str).str.lower().tolist()
train_ids   = train[ID_COL].values
test_ids    = test[ID_COL].values

# ---------- 2. 标题 TF-IDF（char_wb 字符 n-gram，对印尼/英文标题无需分词） ----------
tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 4),       # 3~4 字符 n-gram：能抓「同一商品换说法」的局部重合
    min_df=2,
    max_features=200000,
    sublinear_tf=True,
)
X = tfidf.fit_transform(train_title + test_title)
Xtr = X[:len(train)]          # 训练集（语料库）
Q   = X[len(train):]          # 测试集（查询）

# 查询 → 全部训练样本的直接余弦相似度
S = cosine_similarity(Q, Xtr)          # shape: (20, 训练样本数)

# ---------- 3. 对每个测试样本：直接 Top-K + 2-hop 桥接，取并集 ----------
results = {}
for i in range(len(test_ids)):
    sim = S[i]

    # 3.1 直接命中：与查询最相似的 top-K
    direct = np.argsort(-sim)[:DIRECT_K]

    # 3.2 2-hop 桥接：top-N 种子各自再检索，取「种子→候选」的最大相似度，
    #     把超过阈值、但直接相似度排不进 top-K 的也拉进来
    seeds  = np.argsort(-sim)[:SEED_N]
    bridge = cosine_similarity(Xtr[seeds], Xtr).max(axis=0)

    cand, seen = [], set()
    for o in direct:                       # 先放直接命中的，保证稳定垫底
        seen.add(o); cand.append(o)
    for o in np.argsort(-bridge):          # 再按桥接相似度从高到低并入
        if bridge[o] >= THR and o not in seen:
            seen.add(o); cand.append(o)
        if len(cand) >= MAX_NUM:
            break

    # 3.3 映射回真实 ID（测试 ID 不在训练集里，无自匹配问题）
    ids = [int(train_ids[o]) for o in cand[:MAX_NUM]]
    results[int(test_ids[i])] = ids

# ---------- 4. 输出：Posting_ID, Num, Predicted_Results ----------
out = pd.DataFrame({
    'Posting_ID': list(results.keys()),
    'Num': [len(v) for v in results.values()],
    'Predicted_Results': [','.join(map(str, v)) for v in results.values()],
})
out.to_csv(OUT_PATH, index=False)
print(f'已输出 {len(out)} 行，Num 范围 [{out["Num"].min()}, {out["Num"].max()}]')
print(out.head())
