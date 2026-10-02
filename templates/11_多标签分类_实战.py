# -*- coding: utf-8 -*-
"""
2503 / 202503 网站提问多标签 —— 08 模板读 CSV + 03 模板多标签分类
  · 08 模板:多张 CSV 按主键 merge(问题 Title/Body + 答案 Answer)+ 按 Id 聚合
  · 03 模板做法 B:MultiLabelBinarizer + OneVsRest 逻辑回归,输出逗号分隔 Tag
  · 评分:Micro-F1

修正了 08 模板的两个坑:
  1) 标签用空格 join 但 03 用逗号 split,对不上 → 统一逗号分隔;
  2) group_join 先 drop_duplicates 再聚 Tag,会丢掉多标签 → 标签在聚合时一起逗号合并。
跑法:
    C:\\Users\\yanglang\\anaconda3\\envs\\pytorch\\python.exe 2503_多标签分类.py
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ================= 08 模板:读 CSV + 多表 merge =================
# 四张 CSV 对应原 alldata.db 四张表;按主键 merge 等价于 SQL 的 LEFT JOIN
questions = pd.read_csv('Filtered_Questions.csv', encoding='utf-8-sig')[['Id', 'Title', 'Body']]
answers   = pd.read_csv('Filtered_Answers.csv',  encoding='utf-8-sig')[['ParentId', 'Body']].rename(columns={'Body': 'Answer'})
train_tag = pd.read_csv('Train_data.csv', encoding='utf-8-sig')   # Id, Tag
test_ids  = pd.read_csv('Test_data.csv',  encoding='utf-8-sig')   # Id

# 训练:Train_data ⋈ questions(Id) ⋈ answers(ParentId=Id)
training = (train_tag.merge(questions, on='Id', how='left')
                      .merge(answers, left_on='Id', right_on='ParentId', how='left')
                      .drop(columns='ParentId'))
# 测试:Test_data ⋈ questions ⋈ answers(无标签列)
test = (test_ids.merge(questions, on='Id', how='left')
                 .merge(answers, left_on='Id', right_on='ParentId', how='left')
                 .drop(columns='ParentId'))
print('join 后:training', training.shape, ' test', test.shape)

# 按 Id 聚合:文本列(可能多条答案)拼接去重,标签列逗号分隔去重
def _join(s, sep=' '):
    return sep.join(x for x in s.dropna().astype(str).unique() if x.strip())


def agg_by_id(df):
    agg = df.groupby('Id').agg(
        Title =('Title',  lambda s: _join(s)),
        Body  =('Body',   lambda s: _join(s)),
        Answer=('Answer', lambda s: _join(s)),
    ).reset_index()
    if 'Tag' in df.columns:   # 训练集才有标签
        tags = df.groupby('Id')['Tag'].apply(
            lambda s: ','.join(sorted(set(s.dropna().astype(str))))).reset_index(name='Tag')
        agg = agg.merge(tags, on='Id')
    return agg


training = agg_by_id(training)
test = agg_by_id(test)
print('聚合后:training', training.shape, ' test', test.shape)

# 文本清洗(去 HTML 标签 + 实体 + 小写 + 去标点),Title/Body/Answer 拼成一段
def clean_html(s):
    s = re.sub(r'<[^>]+>', ' ', str(s))       # 去 <p> <a href> 等
    s = re.sub(r'&[a-z#0-9]+;', ' ', s)       # 去 &nbsp; &lt; 等
    s = re.sub(r'[^a-z0-9]+', ' ', s.lower()) # 非字母数字 -> 空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s


training['text'] = (training['Title'].astype(str) + ' ' + training['Body'].astype(str)
                    + ' ' + training['Answer'].astype(str)).apply(clean_html)
test['text'] = (test['Title'].astype(str) + ' ' + test['Body'].astype(str)
                + ' ' + test['Answer'].astype(str)).apply(clean_html)

# ================= 03 模板做法 B:多标签分类 =================
tags_list = [[t.strip() for t in ts.split(',') if t.strip()] for ts in training['Tag']]
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(tags_list)
print('标签数:', len(mlb.classes_))

tf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=30000)
tf.fit(pd.concat([training['text'], test['text']]))
X_tr = tf.transform(training['text'])
X_te = tf.transform(test['text'])

# 验证:留 10% 用于搜阈值 + 对比 Micro-F1
X_a, X_b, y_a, y_b = train_test_split(X_tr, y, test_size=0.1, random_state=42)
cv = OneVsRestClassifier(
    LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear'),
    n_jobs=-1)
cv.fit(X_a, y_a)
proba_val = cv.predict_proba(X_b)

# 每个标签独立搜最优阈值(最大化该标签 F1);低频标签(<10 个正例)固定 0.5 防过拟合
n_tags = y.shape[1]
thr = np.full(n_tags, 0.5)
pos = y_b.sum(axis=0)
for j in range(n_tags):
    if pos[j] < 10:
        continue
    yj, pj = y_b[:, j], proba_val[:, j]
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(yj, (pj >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    thr[j] = best_t

print('① 固定0.5        验证Micro-F1 = %.4f' % f1_score(y_b, (proba_val >= 0.5).astype(int), average='micro'))

# 全局单一阈值:所有标签共用一个 t,直接搜让整体 Micro-F1 最大(严格对准评分)
best_tg, micro_global = 0.5, -1.0
for t in np.arange(0.05, 0.96, 0.01):
    m = f1_score(y_b, (proba_val >= t).astype(int), average='micro')
    if m > micro_global:
        micro_global, best_tg = m, t
print('② 全局阈值 t=%.2f 验证Micro-F1 = %.4f' % (best_tg, micro_global))

pv_opt = (proba_val >= thr[None, :]).astype(int)
print('③ 每标签调优     验证Micro-F1 = %.4f' % f1_score(y_b, pv_opt, average='micro'))
print('   每标签调优 Macro-F1 = %.4f  完全匹配正确率 = %.4f'
      % (f1_score(y_b, pv_opt, average='macro'), (pv_opt == y_b).all(axis=1).mean()))

# 全量训练 + 用调优阈值预测测试集
clf = OneVsRestClassifier(
    LogisticRegression(C=4, max_iter=2000, class_weight='balanced', solver='liblinear'),
    n_jobs=-1)
clf.fit(X_tr, y)
proba = clf.predict_proba(X_te)
p = (proba >= thr[None, :]).astype(int)
for i in range(p.shape[0]):            # 无标签样本兜底:取概率最大那个
    if p[i].sum() == 0:
        p[i, np.argmax(proba[i])] = 1

labels = mlb.inverse_transform(p)
out = pd.DataFrame({'Id': test['Id'], 'Tag': [','.join(ls) for ls in labels]})
out.to_csv('考生姓名_results.csv', index=False)
print(out.head())
