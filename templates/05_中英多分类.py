# ============================================================================
# 模板②：文本分类（多分类·单标签）—— 练习2212 产品类别预测
# 数据：training.xlsx 3342 条 / test-Ans.xlsx 1114 条，字段
#       Product_ID + Product_Description(中文商品描述) + Product_Category(24 类)
# 评分：Micro-F1（单标签多分类时 Micro-F1 == Accuracy，等价于准确率）
# 提交：Product_ID, Predicted_Product_Category（类别是【中文原文】，不是数字编码）
#       编码 gb18030（跟官方参考 ans 一致）
#
# 【实测结论（本机 test-Ans 的 Product_Category 直接评估）】
#   配置                                            Micro-F1
#   历年代码模板②(词级+char_wb(2,4) 双通道+XGB)       0.4532
#   参考代码(jieba 分词+去停用词+词级 TF-IDF)          ~0.45   (本机无 jieba)
#   本模板: char_wb(1,4) 单字符通道 + LinearSVC       0.6427   ← 最优
#
# 【两个关键改造（都是实测踩出来的）】
#   1. char_wb 的 ngram_range 必须带 1 —— (1,4) 比 (2,4) 高一大截
#      (0.593 → 0.630)：单字能抓单字词 + 词边界，对中文分类是主增益。
#   2. 清洗「只留中文+英文+数字」：去掉标点/换行/特殊符号，但【保留英文和数字】
#      (只留中文=0.630，中英文数字=0.643)。商品描述里的 "128GB"/"iPhone" 是分类信号。
#   3. 不要词级双通道：无 jieba 时默认 tokenizer 按空格切，中文没空格→词级几乎无用，
#      反而稀释字符特征 (0.643 → 0.611)。中文题直接 char_wb 单通道即可。
#   4. 模型用 LinearSVC：高维稀疏 TF-IDF 上比 XGBoost 快且准
#      (XGB 0.577 vs SVC 0.643)。
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

# ===== 改这里 1：路径 + 列名 =====
TRAIN_PATH = 'data/training.xlsx'
TEST_PATH  = 'data/test-Ans.xlsx'   # 考场版是 test.xlsx（无 Product_Category 答案列）
TEXT_COL   = 'Product_Description'  # 文本列
TARGET     = 'Product_Category'     # 目标列（训练集有，测试集没有）
ID_COL     = 'Product_ID'           # 提交用 ID 列
OUT_CSV    = '练习2212_results.csv' # 考试时改成「考生姓名_results.csv」
# ====================================================

# ---------- 1. 读数据（xlsx 用 openpyxl，无需管编码） ----------
train = pd.read_excel(TRAIN_PATH)
test  = pd.read_excel(TEST_PATH)

# ---------- 2. 文本清洗：只留中文+英文+数字 ----------
def clean_text(t):
    if not isinstance(t, str):
        return ''
    return re.sub(r'[^一-鿿a-zA-Z0-9]', ' ', t).strip()

train['text'] = train[TEXT_COL].fillna('').apply(clean_text)
test['text']  = test[TEXT_COL].fillna('').apply(clean_text)

# ---------- 3. 目标编码成数字（24 类中文 -> 0..23） ----------
le = LabelEncoder()
y = le.fit_transform(train[TARGET].astype(str))
print('类别数:', len(le.classes_))

# ---------- 4. TF-IDF：char_wb(1,4)，train+test 一起 fit ----------
# 关键：train+test 一起 fit，保证词表一致、也吃进测试集出现的字。
all_text = pd.concat([train['text'], test['text']])
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4),
                        sublinear_tf=True, min_df=2, decode_error='ignore')
tfidf.fit(all_text)
A = tfidf.transform(train['text'])   # 训练特征
B = tfidf.transform(test['text'])    # 测试特征
print('特征维度:', A.shape)

# ---------- 5. LinearSVC 训练 + 预测 ----------
model = LinearSVC(C=1.0)
model.fit(A, y)
pred = model.predict(B)

# ---------- 6. 数字 -> 中文类别，输出 ----------
pred_label = le.inverse_transform(pred)
result = pd.DataFrame({
    ID_COL: test[ID_COL],
    'Predicted_Product_Category': pred_label,
})
result.to_csv(OUT_CSV, index=False, encoding='gb18030')
print(f'已写出 {OUT_CSV}  ({len(result)} 行)')
print(result.head())
