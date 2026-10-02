# -*- coding: utf-8 -*-
# ============================================================================
# 模板③：阅读理解多选题（QA）—— flan-t5 零样本（官方 answer.ipynb 版本6）
# 适用：202406 练习2406 —— article + question + A/B/C/D 四选项，输出正确选项字母
# 评分：Accuracy（提交 question_id + Predicted_Results，值 1~4）
#
# 【核心思路】
#   用 flan-t5-base 零样本做多选题：把 article+question+选项拼成 prompt，
#   让模型直接吐 A/B/C/D 里的一个字母，再映射回 1~4。
#   官方 answer.ipynb 用这套跑到 Accuracy 0.9188。
#
# 【两个必须注意的坑】
#   1. 不要截断到 512：prompt 顺序是「文章在前、问题/选项在后」，naive 截断会
#      把问题和选项整个切掉，答案全丢。T5 用相对位置编码，超 512 只是位置信息
#      饱和、不会报错，官方就是不截断直接跑的。所以这里 truncation=False。
#   2. 模型本地加载（离线考场）：模型在 D:\...\models\flan-t5-base，本地路径加载，
#      不联网、不碰 HF，也不依赖 sentence_transformers，无需 SSL patch。
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ===== 改这里 1：路径 + 列名 =====
MODEL_DIR = r'D:\学习资料\练习\models\flan-t5-base'
train = pd.read_excel('data/training_data.xlsx')
test  = pd.read_excel('data/test_data _label.xlsx')   # 考场版是 test.csv（无 correct_answer）
OUT_CSV = '考生姓名_results.csv'

ID_COL = 'question_id'
OPT_COLS = ['selection1', 'selection2', 'selection3', 'selection4']
TARGET   = 'correct_answer'   # 练习数据里有，考场版没有（下面会自动跳过评分）

# 只跑前 N 条做冒烟测试（None = 全量）
LIMIT = None
# ====================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = (AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, dtype=torch.float16)
         .to(device).eval())

preamble = ('Using the following article to Answer the question by outputting only '
            'one letter from A, B, C, or D that you believe is the most likely to be correct.')


def fmt(r):
    return (f'{preamble}\n\n{str(r["article"])}\n\n'
            f'The question is listed below.\n{str(r["question"])}\n\n'
            f'A) {str(r["selection1"])}\nB) {str(r["selection2"])}\n'
            f'C) {str(r["selection3"])}\nD) {str(r["selection4"])}')


LETTER2IDX = {'A': 1, 'B': 2, 'C': 3, 'D': 4}


def predict_one(text):
    """单条预测：greedy 生成 1 个 token，扫出第一个 A/B/C/D 字母映射回 1~4。"""
    inputs = tokenizer(text, return_tensors='pt', truncation=False).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    s = tokenizer.decode(out[0], skip_special_tokens=True).strip().upper()
    for ch in s:
        if ch in LETTER2IDX:
            return LETTER2IDX[ch]
    return None  # 没扫到字母 -> 交给 fallback


def run(df, desc):
    df = df.reset_index(drop=True)
    if LIMIT is not None:
        df = df.iloc[:LIMIT].reset_index(drop=True)
    preds = []
    for i in tqdm(range(len(df)), desc=desc):
        preds.append(predict_one(fmt(df.iloc[i])))
    # fallback：没输出字母的，填多数类（练习数据里算出来是 3）
    fallback = 3
    preds = [p if p is not None else fallback for p in preds]
    df['Predicted_Results'] = preds
    return df


# ---------- 预测 ----------
res = run(test, 'predict')

# ---------- 练习数据有标签 -> 顺手评个分 ----------
if TARGET in res.columns:
    y_true = res[TARGET].astype(int)
    y_pred = res['Predicted_Results'].astype(int)
    from sklearn.metrics import accuracy_score, classification_report
    print('Accuracy: %.4f' % accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# ---------- 输出提交文件 ----------
res[[ID_COL, 'Predicted_Results']].to_csv(OUT_CSV, index=False)
print('已写出', OUT_CSV)
print(res[[ID_COL, 'Predicted_Results']].head(10))
