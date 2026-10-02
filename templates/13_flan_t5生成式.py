# -*- coding: utf-8 -*-
# ============================================================================
# 2509 问答判定 —— flan-t5 生成式解法(官方解思路)
#   任务:question + passage -> 判断 passage 能否回答 question(True/False)
#   做法:把「文章 + 问题」喂给 flan-t5,让它直接生成 True/False
#   依赖:本地模型 models/flan-t5-base(已离线下载),GPU 加速
# ============================================================================
import warnings
warnings.filterwarnings('ignore')

# 坑:Windows 坏证书 -> patch ssl(加载本地模型一般用不到,但保险)
import ssl
_orig = ssl.SSLContext.load_default_certs
def _patched(self, *args, **kwargs):
    try:
        _orig(self, *args, **kwargs)
    except Exception:
        pass
ssl.SSLContext.load_default_certs = _patched

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = r'D:/学习资料/练习/models/flan-t5-base'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device:', device)

tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
model.eval()

# 参考解用的 prompt:文章 + 问题 -> 输出 True/False
PREAMBLE = ('Use the articles about the question, '
            'Answer the following question by outputting the True or False')

def make_prompt(passage, question):
    return f'{PREAMBLE}\n\n articles : {passage}\n\n question : {question}'

def parse(ans):
    a = str(ans).strip().lower()
    if 'true' in a or 'yes' in a:
        return True
    if 'false' in a or 'no' in a:
        return False
    return True   # 兜底:多数类

@torch.no_grad()
def predict_batch(prompts, batch=16):
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i+batch], return_tensors='pt', padding=True,
                  truncation=True, max_length=512).to(device)
        gen = model.generate(**enc, max_new_tokens=5, do_sample=False)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out

# ===== 读数据(CSV,UTF-8 混坏字节)=====
train = pd.read_csv('train.csv', encoding='utf-8', encoding_errors='replace')
test  = pd.read_csv('test.csv',  encoding='utf-8', encoding_errors='replace')

# ===== 先在训练集上验证,确认能超过多数类基线(78%)=====
print('验证集(训练数据)上跑 flan-t5 ...')
train_prompts = [make_prompt(r.passage, r.question) for r in train.itertuples()]
preds = [parse(a) for a in predict_batch(train_prompts)]
y = train['Predicted_Results'].astype(bool)
acc = np.mean(np.array(preds) == y.values)
print('训练集准确率 = %.4f  (多数类基线 = %.4f)' % (acc, y.mean()))

# ===== 预测测试集 =====
print('预测测试集 %d 条 ...' % len(test))
test_prompts = [make_prompt(r.passage, r.question) for r in test.itertuples()]
test_preds = [parse(a) for a in predict_batch(test_prompts)]

out = pd.DataFrame({'question_id': test['question_id'],
                    'Predicted_Results': np.where(test_preds, 'TRUE', 'FALSE')})
out.to_csv('考生姓名_results.csv', index=False)
print('输出分布:\n', out['Predicted_Results'].value_counts())
